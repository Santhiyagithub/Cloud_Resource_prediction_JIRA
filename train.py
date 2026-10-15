"""
train.py
--------
Cloud Resource Prediction — Model Training and MLflow Experiment Tracking
=========================================================================
Dataset schema: Alibaba Cluster Trace 2017 (machine_usage table)
  Source: https://github.com/alibaba/clusterdata
  Columns: machine_id, time_stamp, cpu_util, mem_util, net_in, net_out, disk_io

NOTE: The dataset used here is SYNTHETIC, mirroring the Alibaba schema.
      The full Alibaba trace requires a separate download/registration.
      This must be disclosed in the project report.

This script:
  1. Loads cloud_data.csv (Alibaba schema).
  2. Engineers lag features for time-series forecasting.
  3. Evaluates a persistence (last-value) baseline.
  4. Trains three Random Forest configurations and logs all runs to MLflow.
  5. Saves the best model (lowest RMSE) to models/model.joblib.
"""

import os
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ---------------------------------------------------------------------------
# 1. Load and validate data
# ---------------------------------------------------------------------------
DATA_PATH  = os.path.join("data", "cloud_data.csv")
MODEL_PATH = os.path.join("models", "model.joblib")

print("=" * 60)
print("  Cloud Resource Prediction - Training Script")
print("  Dataset schema: Alibaba Cluster Trace 2017")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[Data] Loaded {len(df)} rows from {DATA_PATH}")
print(f"[Data] Machines: {df['machine_id'].unique().tolist()}")

# Aggregate across machines: average utilization per time_stamp
# (simulates per-cluster aggregated view, standard for forecasting)
df_agg = (
    df.groupby("time_stamp")
      .agg(
          cpu_util=("cpu_util",  "mean"),
          mem_util=("mem_util",  "mean"),
          net_in  =("net_in",    "mean"),
          net_out =("net_out",   "mean"),
          disk_io =("disk_io",   "mean"),
      )
      .reset_index()
      .sort_values("time_stamp")
)

# Filter physically plausible rows
df_agg = df_agg[
    df_agg["cpu_util"].between(0, 100) &
    df_agg["mem_util"].between(0, 100)
].dropna()

print(f"\n[Data] After aggregation: {len(df_agg)} timesteps")
print(df_agg[["cpu_util", "mem_util", "net_in", "net_out"]].describe().round(2))

# ---------------------------------------------------------------------------
# 2. Feature engineering (lag features)
# ---------------------------------------------------------------------------
df_agg["cpu_lag1"] = df_agg["cpu_util"].shift(1)   # t-1
df_agg["cpu_lag2"] = df_agg["cpu_util"].shift(2)   # t-2
df_agg["mem_lag1"] = df_agg["mem_util"].shift(1)   # memory t-1
df_agg["target"]   = df_agg["cpu_util"].shift(-1)  # predict t+1 CPU
df_agg = df_agg.dropna()

FEATURES = ["cpu_util", "mem_util", "net_in", "net_out",
            "cpu_lag1", "cpu_lag2", "mem_lag1"]

# ---------------------------------------------------------------------------
# 3. Chronological train/test split (no shuffling — time-series)
# ---------------------------------------------------------------------------
SPLIT_RATIO = 0.80
split_idx   = int(len(df_agg) * SPLIT_RATIO)
train = df_agg.iloc[:split_idx]
test  = df_agg.iloc[split_idx:]

X_train, y_train = train[FEATURES], train["target"]
X_test,  y_test  = test[FEATURES],  test["target"]

print(f"\n[Split] Train: {len(train)} rows | Test: {len(test)} rows")

# ---------------------------------------------------------------------------
# 4. Helper
# ---------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return round(mae, 4), round(rmse, 4)

# ---------------------------------------------------------------------------
# 5. MLflow experiment
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "cloud-resource-prediction"
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"\n[MLflow] Experiment: '{EXPERIMENT_NAME}'")

results = []

# --- Run 1: Persistence Baseline ---
with mlflow.start_run(run_name="baseline_persistence"):
    y_pred_base = X_test["cpu_util"]
    mae, rmse   = compute_metrics(y_test, y_pred_base)

    mlflow.log_param("model",    "persistence_baseline")
    mlflow.log_param("strategy", "last_known_cpu_util")
    mlflow.log_param("dataset",  "Alibaba-schema synthetic")
    mlflow.log_param("features", "cpu_util (last value only)")
    mlflow.log_metric("mae",  mae)
    mlflow.log_metric("rmse", rmse)

    results.append({"run": "baseline", "n_est": "-", "max_d": "-",
                    "mae": mae, "rmse": rmse})
    print(f"\n  [Baseline]  MAE = {mae:.4f}  |  RMSE = {rmse:.4f}")

# --- Runs 2-4: Random Forest ---
RF_CONFIGS = [(50, 5), (100, 10), (200, 15)]
best_rmse  = float("inf")
best_model = None
best_config = None

for n_est, max_d in RF_CONFIGS:
    run_name = f"rf_n{n_est}_d{max_d}"
    with mlflow.start_run(run_name=run_name):
        model = RandomForestRegressor(
            n_estimators=n_est,
            max_depth=max_d,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)
        mae, rmse = compute_metrics(y_test, y_pred)

        mlflow.log_param("model",        "RandomForestRegressor")
        mlflow.log_param("n_estimators", n_est)
        mlflow.log_param("max_depth",    max_d)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("dataset",      "Alibaba-schema synthetic")
        mlflow.log_param("features",     str(FEATURES))
        mlflow.log_metric("mae",  mae)
        mlflow.log_metric("rmse", rmse)

        # MLflow 3.x: use name= and declare skops trusted types
        mlflow.sklearn.log_model(
            model,
            name="model",
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        results.append({"run": run_name, "n_est": n_est,
                        "max_d": max_d, "mae": mae, "rmse": rmse})
        print(f"  [{run_name}]  n={n_est:>3}  d={max_d:>2}  "
              f"MAE = {mae:.4f}  |  RMSE = {rmse:.4f}")

        if rmse < best_rmse:
            best_rmse   = rmse
            best_model  = model
            best_config = (n_est, max_d)

# ---------------------------------------------------------------------------
# 6. Save best model
# ---------------------------------------------------------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, MODEL_PATH)
print(f"\n[Model] Best: n_estimators={best_config[0]}, "
      f"max_depth={best_config[1]}  (RMSE={best_rmse:.4f})")
print(f"[Model] Saved -> {MODEL_PATH}")

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  EXPERIMENT SUMMARY")
print("=" * 60)
print(f"{'Run':<22} {'n_est':>5} {'max_d':>5} {'MAE':>8} {'RMSE':>8}")
print("-" * 55)
for r in results:
    print(f"{r['run']:<22} {str(r['n_est']):>5} {str(r['max_d']):>5} "
          f"{r['mae']:>8.4f} {r['rmse']:>8.4f}")
print("=" * 60)
print("\nNext steps:")
print("  mlflow ui --host 127.0.0.1 --port 5000")
print("  python app.py")
