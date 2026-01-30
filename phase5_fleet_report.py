import pandas as pd

# Load the saved validation decisions
df = pd.read_csv("data/processed/val_predictions_decisions.csv")

# Keep only the latest cycle per engine (fleet snapshot)
fleet = (
    df.sort_values(["engine_id", "cycle"])
      .groupby("engine_id")
      .tail(1)
      .copy()
)

# Risk ranking: Service first, then Monitor, then OK; inside each sort by predicted RUL asc
decision_order = {"SERVICE_NOW": 0, "MONITOR": 1, "OK": 2}
fleet["decision_rank"] = fleet["decision"].map(decision_order)

fleet = fleet.sort_values(["decision_rank", "RUL_pred", "health_index"], ascending=[True, True, True])

print("=== FLEET SNAPSHOT (Latest cycle per engine) ===")
print(fleet[["engine_id", "cycle", "RUL_pred", "health_index", "decision"]].head(20))

# Save report
fleet.drop(columns=["decision_rank"]).to_csv("data/processed/fleet_snapshot.csv", index=False)
print("\nSaved: data/processed/fleet_snapshot.csv")
