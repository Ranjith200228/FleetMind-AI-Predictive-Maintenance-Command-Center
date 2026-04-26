"""End-to-end training + evaluation on a C-MAPSS subset (FD001 or FD003).

Runs:
  1. Load the subset, add piecewise RUL, fit MinMaxScaler on train
  2. Train Random Forest on engineered features
  3. Train LSTM on windowed sequences
  4. Evaluate both on the held-out test engines
  5. Write reports/metrics_<SUBSET>.json with comparison vs. published baselines
     (and reports/metrics.json as a legacy alias for FD001)

Usage:
    python -m scripts.train_and_eval                  # FD001, default
    python -m scripts.train_and_eval --subset FD003   # FD003
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.metrics import score_report  # noqa: E402
from src.preprocess import (  # noqa: E402
    RUL_CAP,
    SUPPORTED_SUBSETS,
    add_piecewise_rul,
    feature_cols_for,
    fit_scaler,
    load_fd,
    make_sequences,
    make_test_sequences,
)
from src.rf_baseline import predict_rf_last, save_rf, train_rf  # noqa: E402
from src.train_lstm import train_lstm  # noqa: E402

# Per-subset published baselines for comparison (RMSE / Score on test set).
# FD003 is harder for early methods (two fault modes), easier for modern ones.
PUBLISHED_BASELINES: dict[str, list[dict]] = {
    "FD001": [
        {"name": "MLP (Babu et al., 2016)",       "rmse": 37.56, "score": 17972},
        {"name": "SVR (Babu et al., 2016)",       "rmse": 20.96, "score": 1381},
        {"name": "CNN (Babu et al., 2016)",       "rmse": 18.45, "score": 1287},
        {"name": "LSTM (Zheng et al., 2017)",     "rmse": 16.14, "score": 338},
        {"name": "Deep LSTM (Wu et al., 2018)",   "rmse": 13.65, "score": 280},
        {"name": "BiLSTM (Wang et al., 2018)",    "rmse": 13.65, "score": 295},
        {"name": "AGCNN (Liu et al., 2020)",      "rmse": 12.42, "score": 226},
    ],
    "FD003": [
        {"name": "MLP (Babu et al., 2016)",       "rmse": 37.39, "score": 17409},
        {"name": "CNN (Babu et al., 2016)",       "rmse": 19.82, "score": 1596},
        {"name": "LSTM (Zheng et al., 2017)",     "rmse": 16.18, "score":  852},
        {"name": "Deep LSTM (Wu et al., 2018)",   "rmse": 16.18, "score": 1370},
        {"name": "BiLSTM (Wang et al., 2018)",    "rmse": 13.74, "score":  317},
        {"name": "AGCNN (Liu et al., 2020)",      "rmse": 12.51, "score":  264},
    ],
}

BASELINE_SOURCES = [
    "Babu, Zhao, Li. 'Deep CNN approach for estimation of RUL of machinery.' DASFAA 2016.",
    "Zheng, Ristovski, Farahat, Gupta. 'Long short-term memory network for RUL estimation.' ICPHM 2017.",
    "Wu, Yuan, Dong, Lin. 'Remaining useful life estimation with deep LSTM.' IEEE Trans. Ind. Inf. 2018.",
    "Wang, Wu, Yu, Xu. 'Remaining useful life estimation based on BiLSTM.' ICMNE 2018.",
    "Liu, Song, Pan, Zio. 'Attention-based gated CNN for RUL prediction.' RESS 2020.",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--subset", default="FD001",
                   choices=list(SUPPORTED_SUBSETS),
                   help="C-MAPSS subset to train on")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--window", type=int, default=30)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--reports-dir", default="reports")
    p.add_argument("--models-dir", default="models")
    p.add_argument("--skip-lstm", action="store_true")
    p.add_argument("--skip-rf", action="store_true")
    args = p.parse_args()

    subset = args.subset.upper()
    sub_lc = subset.lower()
    feature_cols = feature_cols_for(subset)

    t0 = time.time()
    print(f"[load] reading {subset} from {args.data_dir}")
    train, test, rul_true = load_fd(subset, args.data_dir)
    train = add_piecewise_rul(train, cap=RUL_CAP)
    print(f"[load] train={len(train):,} rows / {train['engine_id'].nunique()} engines | "
          f"test={len(test):,} rows / {test['engine_id'].nunique()} engines | "
          f"features={len(feature_cols)} ({feature_cols[0]}..{feature_cols[-1]})")

    scaler = fit_scaler(train, feature_cols)
    reports_dir = Path(args.reports_dir)
    models_dir = Path(args.models_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"scaler": scaler, "feature_cols": feature_cols,
                 "window": args.window, "rul_cap": RUL_CAP, "subset": subset},
                models_dir / f"preprocess_{sub_lc}.joblib")

    rul_true_capped = np.minimum(rul_true.values.astype(np.float32), RUL_CAP)

    results: dict = {
        "dataset": f"C-MAPSS {subset}",
        "subset": subset,
        "rul_cap": RUL_CAP,
        "window": args.window,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        f"published_baselines_{sub_lc}": PUBLISHED_BASELINES[subset],
        "baseline_sources": BASELINE_SOURCES,
    }

    if not args.skip_rf:
        print("[rf] building features and training...")
        t_rf0 = time.time()
        rf_model, rf_cols = train_rf(train, feature_cols,
                                     window=args.window, seed=args.seed)
        save_rf(rf_model, rf_cols, models_dir / f"rf_{sub_lc}.joblib")
        rf_pred, rf_eids = predict_rf_last(rf_model, test, feature_cols,
                                            rf_cols, window=args.window)
        rf_pred = np.clip(rf_pred, 0, RUL_CAP)
        rf_metrics = score_report(rul_true_capped, rf_pred)
        rf_metrics["train_seconds"] = round(time.time() - t_rf0, 1)
        results["random_forest"] = rf_metrics
        print(f"[rf] RMSE={rf_metrics['rmse']:.2f}  MAE={rf_metrics['mae']:.2f}  "
              f"Score={rf_metrics['cmapss_score']:.1f}")

    if not args.skip_lstm:
        print("[lstm] windowing sequences...")
        X_tr, y_tr, _ = make_sequences(train, feature_cols, scaler,
                                        window=args.window)
        X_te, te_eids = make_test_sequences(test, feature_cols, scaler,
                                             window=args.window)
        print(f"[lstm] X_train={X_tr.shape}  X_test={X_te.shape}")
        t_l0 = time.time()
        model = train_lstm(
            X_tr, y_tr, window=args.window,
            out_path=models_dir / f"lstm_{sub_lc}.keras",
            epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
        )
        lstm_pred = model.predict(X_te, verbose=0).ravel()
        lstm_pred = np.clip(lstm_pred, 0, RUL_CAP)
        lstm_metrics = score_report(rul_true_capped, lstm_pred)
        lstm_metrics["train_seconds"] = round(time.time() - t_l0, 1)
        lstm_metrics["params"] = int(model.count_params())
        results["lstm"] = lstm_metrics
        print(f"[lstm] RMSE={lstm_metrics['rmse']:.2f}  MAE={lstm_metrics['mae']:.2f}  "
              f"Score={lstm_metrics['cmapss_score']:.1f}")
        # dump per-engine predictions for the UI
        pd.DataFrame({
            "engine_id": te_eids,
            "true_rul": rul_true_capped,
            "lstm_pred": lstm_pred,
        }).to_csv(reports_dir / f"{sub_lc}_predictions.csv", index=False)

    results["elapsed_seconds"] = round(time.time() - t0, 1)
    out_per_subset = reports_dir / f"metrics_{subset}.json"
    out_per_subset.write_text(json.dumps(results, indent=2))
    print(f"[done] wrote {out_per_subset}  ({results['elapsed_seconds']}s total)")

    # Legacy alias: FD001 also writes the original reports/metrics.json so
    # any earlier consumers (and the existing dashboard) keep working.
    if subset == "FD001":
        (reports_dir / "metrics.json").write_text(json.dumps(results, indent=2))
        print(f"[done] also wrote reports/metrics.json (legacy alias)")


if __name__ == "__main__":
    main()
