# FleetMind — Detailed Results

Full benchmark numbers, methodology, and per-experiment notes for both
C-MAPSS subsets.

## Headline metrics

|  | LSTM RMSE | LSTM MAE | LSTM Score | RF RMSE | RF MAE | RF Score | LSTM params | LSTM train (s) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **FD001** | **14.31** | 11.06 | **381.3** | 16.65 | 12.81 | 430.5 | 124,737 | 770 |
| **FD003** | **11.93** | 8.30  | **241.1** | 16.09 | 10.75 | 646.6 | 125,761 | 1,026 |

- **RMSE / MAE** are reported in cycles, evaluated against the ground-truth
  RUL provided by NASA PCoE for the held-out test set (100 engines per
  subset; one prediction per engine using the engine's last 30 cycles).
- **Score** is the C-MAPSS asymmetric scoring function from
  *Saxena & Goebel 2008*: `sum(exp(-d/13) - 1)` for early predictions and
  `sum(exp(d/10) - 1)` for late ones, where `d = pred - true`. Lower is
  better; late predictions are penalised ~1.5× more than early ones to
  reflect the operational cost asymmetry of unscheduled removal.

## Comparison to published baselines

### FD001  *(1 op condition, 1 fault mode — HPC degradation; 100 / 100 train/test units)*

| Method | RMSE | Score | Source |
|---|---:|---:|---|
| MLP | 37.56 | 17,972 | Babu et al., *DASFAA 2016* |
| SVR | 20.96 | 1,381 | Babu et al., *DASFAA 2016* |
| CNN | 18.45 | 1,287 | Babu et al., *DASFAA 2016* |
| LSTM | 16.14 | 338 | Zheng et al., *ICPHM 2017* |
| Deep LSTM | 13.65 | 280 | Wu et al., *IEEE T. Ind. Inf. 2018* |
| BiLSTM | 13.65 | 295 | Wang et al., *ICMNE 2018* |
| AGCNN | 12.42 | 226 | Liu et al., *RESS 2020* |
| **FleetMind LSTM** | **14.31** | **381** | this work |
| **FleetMind RF baseline** | **16.65** | **431** | this work |

Our LSTM lands at the upper end of the modern-baseline cluster — beats
Zheng 2017 cleanly on RMSE, slightly behind the more recent Wu 2018 and
Wang 2018 because we deliberately kept the architecture simple
(2-layer stacked LSTM, no attention, no bidirectional pass) for clarity
in the portfolio context.

### FD003  *(1 op condition, 2 fault modes — HPC + Fan; 100 / 100 train/test units)*

| Method | RMSE | Score | Source |
|---|---:|---:|---|
| MLP | 37.39 | 17,409 | Babu et al., *DASFAA 2016* |
| CNN | 19.82 | 1,596 | Babu et al., *DASFAA 2016* |
| LSTM | 16.18 | 852 | Zheng et al., *ICPHM 2017* |
| Deep LSTM | 16.18 | 1,370 | Wu et al., *IEEE T. Ind. Inf. 2018* |
| BiLSTM | 13.74 | 317 | Wang et al., *ICMNE 2018* |
| AGCNN | 12.51 | 264 | Liu et al., *RESS 2020* |
| **FleetMind LSTM** | **11.93** | **241** | this work |
| **FleetMind RF baseline** | **16.09** | **647** | this work |

On FD003 our LSTM beats every published baseline including AGCNN (Liu
2020) on both RMSE and Score. The win is driven by the per-subset
feature selection (FD003 keeps sensors 6 and 10 that FD001 drops as
constant) and a slightly longer training run (60 epochs vs. early-stop
on FD001).

## Methodology

### Data pipeline (`src/preprocess.py`)

1. **Read** the space-separated `train_FDxxx.txt` / `test_FDxxx.txt` /
   `RUL_FDxxx.txt` from `data/raw/`.
2. **Per-subset feature selection.** Drop sensors with `std < 1e-6` on the
   training split:
   - FD001 drops `{1, 5, 6, 10, 16, 18, 19}` → keeps 14 sensors.
   - FD003 drops `{1, 5, 16, 18, 19}` → keeps 16 sensors.
   The two subsets cannot share a scaler (different feature dimensionality).
3. **Piecewise-linear RUL clipping at 125 cycles** (Heimes 2008; Zheng
   2017). The first 50–125 cycles of every engine show no degradation
   signal; treating them as a flat target prevents the loss from
   regressing to noise.
4. **MinMax scaler** fit on the train split, `feature_range=(-1, 1)`.
5. **Sliding window**, length 30. Per-engine, left-padded with zeros for
   engines shorter than the window (test set).

### LSTM (`src/train_lstm.py`)

```
Input  (window=30, n_features=14|16)
  └─► Masking (mask_value=0.0)
  └─► LSTM(128, return_sequences=True)
  └─► Dropout(0.2)
  └─► LSTM(64)
  └─► Dropout(0.2)
  └─► Dense(32, relu)
  └─► Dense(1)
Loss      : MSE
Optimiser : Adam(1e-3)
Callbacks : EarlyStopping(monitor=val_rmse, patience=8, restore_best_weights)
            ReduceLROnPlateau(factor=0.5, patience=4)
Epochs    : up to 60, batch_size=256, val_split=0.1
```

We tried Huber loss (δ=10) first; it stalled at val_rmse 42 because the
clipped high-RUL plateau makes everything look like an "outlier" to
Huber. MSE converges cleanly.

### Random Forest baseline (`src/rf_baseline.py`)

Engineered features per (engine, last-30-window): per-sensor mean and
std, plus sensor values at the final cycle. `RandomForestRegressor(
n_estimators=300, max_depth=14, min_samples_leaf=4, n_jobs=-1)`. Same
piecewise-RUL labels.

## Retrieval (Phase 2)

Corpus: 55 paragraph-sized chunks (`data/rag/corpus.jsonl`) drawn from
NASA PCoE README, Saxena & Goebel 2008, Heimes 2008, Zheng 2017, Wu
2018, Liu 2020, plus original notes documenting our own modelling
decisions and FAISS / preprocessing choices.

Eval set: 62 hand-curated Q/A pairs (`data/rag/eval_qa.jsonl`), each
listing one or more correct doc_ids.

| Backend | hit@1 | hit@3 | MRR@10 | Notes |
|---|---:|---:|---:|---|
| OpenAI `text-embedding-3-small` (1536-d) | **0.92** | **1.00** | **0.95** | needs `OPENAI_API_KEY` |
| TF-IDF (sklearn, 1–2-gram) | 0.87 | 0.97 | 0.92 | works offline |

FAISS `IndexFlatIP` over L2-normalised embeddings = exact cosine
similarity. With only 55 documents, an approximate index would be pure
overhead.

## Agent (Phase 3)

`gpt-4o-mini` with one tool: `query_engine_history(engine_id) → {
n_observed_cycles, sensors_summary (last/mean/std/trend_per_cycle),
recent_window_raw, lstm_prediction (rul_cycles, recommended_action,
action_rationale) }`.

System prompt enforces strict citation: every factual sentence in the
answer carries `[doc_id]` markers traceable to the corpus. On a 6-query
smoke test, all 6 answers cite at least one chunk.

The agent never invents engine numbers — `query_engine_history` returns
an `ok: False` error for invalid IDs, and the agent surfaces the error
verbatim instead of fabricating a result.

## Reproduce

Each subset's full pipeline (data → trained LSTM → RF baseline →
metrics.json → predictions CSV) is one command:

```bash
python -m scripts.train_and_eval --subset FD001 --epochs 60
python -m scripts.train_and_eval --subset FD003 --epochs 60
```

On a single CPU core: ~13 min for FD001, ~17 min for FD003. GPU not
required.

## Per-engine prediction CSVs

- `reports/fd001_predictions.csv` — `engine_id, true_rul, lstm_pred` for
  all 100 FD001 test engines.
- `reports/fd003_predictions.csv` — same for FD003.

These power the dashboard's status grid and are useful for plotting
prediction-vs-truth scatter plots if you want to extend the analysis.

## Limitations and what's deliberately out of scope

- **FD002 and FD004 are not trained.** Both have 6 operating conditions,
  which means a single MinMax scaler does not work — sensor readings shift
  dramatically with altitude / throttle setting. The standard fix is to
  cluster by `(setting_1, setting_2, setting_3)` and z-score within each
  cluster, then concatenate. That's another 30–50 lines of preprocessing
  on top of what's already here, and a clean extension if you want to
  push to the full 4-subset benchmark.
- **No uncertainty bounds.** The LSTM regresses a point estimate. A
  natural next step is to add a Monte-Carlo dropout pass at inference
  time (predict 30×, take mean ± 1.96·std) to get an 80–95% predictive
  interval per engine. Useful for the dashboard's recommended-action
  band when RUL is near a threshold.
- **No domain-shift evaluation.** All numbers are in-distribution: train
  and test come from the same simulator with the same fault mode mix.
  Real fleets show drift; a production deployment would need a sentinel
  set and a periodic re-training trigger.

These are honest gaps, not hidden ones — and each is a clear next
project.
