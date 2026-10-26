"""
====================================================================
FASAL SURAKSHA - train_yield.py (Revised & Optimized)
Agronomic Yield Forecasting Model with Monotone Constraints & Uncertainty Ranges.

MODES:
1) REAL FIELD DATA:
     python train_yield.py --csv field_data.csv --unit t_per_ha --group_col farm_id

2) SYNTHETIC AGRONOMIC DEMO:
     python train_yield.py --demo --unit t_per_acre

IMPROVEMENTS OVER LEGACY VERSION:
- Monotone Constraints: Disease severity has a negative monotone constraint (-1),
  guaranteeing that increased disease damage can NEVER result in higher yield.
- Dynamic Counterfactual Baseline: Eliminates hardcoded 24 t/acre fallback;
  the healthy baseline is computed dynamically for the same field at 0% severity.
- Grouped Cross-Validation: Splits by farm/group (GroupKFold / GroupShuffleSplit)
  to prevent plot-level data leakage.
- 80% Prediction Ranges: Computes q10 and q90 residual quantiles from cross-validation.
- Out-of-Distribution Tracking: Records feature min/max bounds to alert if inputs exceed trained ranges.
====================================================================
"""
import argparse
import os
from datetime import datetime, timezone
from typing import Optional, Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from xgboost import XGBRegressor

FEATURES = ["crop_age_days", "rainfall_mm", "soil_ph", "temperature_c", "disease_severity_percent"]
SEVERITY_COL = "disease_severity_percent"

VALID_RANGES = {
    "crop_age_days": (1.0, 400.0),
    "rainfall_mm": (0.0, 6000.0),
    "soil_ph": (3.0, 10.0),
    "temperature_c": (-5.0, 55.0),
    "disease_severity_percent": (0.0, 100.0),
}
MIN_ROWS_HARD = 50
MIN_ROWS_SOFT = 300


# ─────────────────────────────────────────────────────────────────
# 1. Dataset Loaders & Synthetic Generator
# ─────────────────────────────────────────────────────────────────
def load_real_csv(path: str, target: str = "actual_yield", group_col: str = "farm_id") -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Loads real farm harvest CSV, validates bounds, and extracts group IDs."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")

    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [target] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    n0 = len(df)
    cols = FEATURES + [target] + ([group_col] if group_col in df.columns else [])
    df = df[cols].copy()

    for c in FEATURES + [target]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    for c, (lo, hi) in VALID_RANGES.items():
        df = df[(df[c] >= lo) & (df[c] <= hi)]
    df = df[df[target] > 0]

    logger.info(f"Loaded {n0} rows, retained {len(df)} after validation and boundary filtering.")

    groups = df[group_col].astype(str).values if group_col in df.columns else None
    if groups is None:
        logger.warning(f"No '{group_col}' column found. Standard KFold will be used instead of GroupKFold.")

    return df.rename(columns={target: "target"}).drop(columns=[group_col], errors="ignore"), groups


def make_demo_data(n: int = 4000, base_yield: float = 24.0, seed: int = 42) -> Tuple[pd.DataFrame, Optional[np.ndarray]]:
    """Generates synthetic agronomic dataset grounded in physiological response curves."""
    r = np.random.RandomState(seed)
    age = r.uniform(30.0, 130.0, n)
    rain = r.uniform(250.0, 1200.0, n)
    ph = r.uniform(5.2, 8.2, n)
    temp = r.uniform(18.0, 38.0, n)
    sev = r.uniform(0.0, 95.0, n)

    g = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2)

    maturity_factor = 0.3 + 0.7 / (1.0 + np.exp(-(age - 60.0) / 12.0))
    ph_factor = 0.6 + 0.4 * g(ph, 6.5, 0.8)
    rain_factor = 0.6 + 0.4 * g(rain, 650.0, 280.0)
    temp_factor = 0.7 + 0.3 * g(temp, 26.0, 6.0)
    disease_damage = 1.0 - 0.85 * (sev / 100.0) ** 1.3

    y = base_yield * maturity_factor * ph_factor * rain_factor * temp_factor * disease_damage
    y = np.clip(y + r.normal(0, 0.04 * base_yield, n), 0.05 * base_yield, 1.25 * base_yield)

    df = pd.DataFrame({
        "crop_age_days": age.round(1),
        "rainfall_mm": rain.round(1),
        "soil_ph": ph.round(2),
        "temperature_c": temp.round(1),
        SEVERITY_COL: sev.round(1),
        "target": y.round(2)
    })
    return df, None


# ─────────────────────────────────────────────────────────────────
# 2. Model Architecture & Cross-Validation
# ─────────────────────────────────────────────────────────────────
def make_model(seed: int = 42) -> XGBRegressor:
    """Builds XGBoost Regressor with negative monotone constraint on disease severity."""
    # -1 means as disease severity increases, predicted yield must never increase
    mono = tuple(-1 if f == SEVERITY_COL else 0 for f in FEATURES)
    return XGBRegressor(
        n_estimators=300,
        learning_rate=0.04,
        max_depth=4,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=4.0,
        monotone_constraints=mono,
        random_state=seed,
        n_jobs=-1
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
    }


def out_of_fold_evaluation(X: pd.DataFrame, y: pd.Series, groups: Optional[np.ndarray], seed: int = 42, k: int = 5):
    """Executes out-of-fold cross validation with group independence."""
    if groups is not None and len(set(groups)) >= k:
        splits = GroupKFold(n_splits=k).split(X, y, groups)
    else:
        splits = KFold(n_splits=k, shuffle=True, random_state=seed).split(X, y)

    oof = np.zeros(len(y))
    base = np.zeros(len(y))
    lin = np.zeros(len(y))

    for tr, te in splits:
        model = make_model(seed).fit(X.iloc[tr], y.iloc[tr])
        oof[te] = model.predict(X.iloc[te])
        base[te] = y.iloc[tr].mean()
        lin[te] = LinearRegression().fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])

    return oof, base, lin


# ─────────────────────────────────────────────────────────────────
# 3. Training & Packaging Pipeline
# ─────────────────────────────────────────────────────────────────
def train(
    df: pd.DataFrame,
    groups: Optional[np.ndarray],
    unit: str = "t_per_acre",
    data_source: str = "synthetic_demo",
    out_path: str = "models/yield_xgboost.pkl",
    seed: int = 42
) -> Dict[str, Any]:
    n = len(df)
    if n < MIN_ROWS_HARD:
        raise ValueError(f"Insufficient samples: {n} rows. Need at least {MIN_ROWS_HARD} rows.")

    X = df[FEATURES].reset_index(drop=True)
    y = df["target"].reset_index(drop=True)
    g = None if groups is None else np.asarray(groups)

    # 1. Hold-out 15% for blind verification
    idx = np.arange(n)
    if g is not None and len(set(g)) >= 10:
        dev_i, hold_i = next(GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed).split(X, y, g))
    else:
        dev_i, hold_i = train_test_split(idx, test_size=0.15, random_state=seed)

    Xd, yd = X.iloc[dev_i].reset_index(drop=True), y.iloc[dev_i].reset_index(drop=True)
    Xh, yh = X.iloc[hold_i].reset_index(drop=True), y.iloc[hold_i].reset_index(drop=True)
    gd = None if g is None else g[dev_i]

    # 2. Cross-validation on dev set
    oof, base, lin = out_of_fold_evaluation(Xd, yd, gd, seed=seed)
    cv = {
        "xgboost": compute_metrics(yd, oof),
        "linear_regression": compute_metrics(yd, lin),
        "guess_the_average": compute_metrics(yd, base)
    }

    # Error quantiles for 80% range interval [10th to 90th percentile]
    resid = yd.values - oof
    q10, q90 = (float(v) for v in np.quantile(resid, [0.10, 0.90]))

    # 3. Hold-out evaluation
    hold_model = make_model(seed).fit(Xd, yd)
    hold_pred = hold_model.predict(Xh)
    holdout = compute_metrics(yh, hold_pred)
    in_range = (yh.values >= hold_pred + q10) & (yh.values <= hold_pred + q90)
    holdout["range_coverage_80pct_actual"] = round(float(np.mean(in_range)), 3)

    # 4. Final model fitted on complete dataset
    final_model = make_model(seed).fit(X, y)

    payload = {
        "model": final_model,
        "features": FEATURES,
        "unit": unit,
        "data_source": data_source,
        "n_rows": int(n),
        "n_groups": None if g is None else int(len(set(g))),
        "train_ranges": {f: [float(X[f].min()), float(X[f].max())] for f in FEATURES},
        "residual_q10": q10,
        "residual_q90": q90,
        "metrics": {"cross_validation": cv, "holdout": holdout},
        "baseline_potential_yield_tons": float(y.max()),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    joblib.dump(payload, out_path)

    logger.info(f"Model saved to {out_path}")
    logger.info(f"Cross-Validation R²: {cv['xgboost']['r2']} | Holdout R²: {holdout['r2']}")
    logger.info(f"80% Interval: {q10:+.2f} to {q90:+.2f} {unit}")
    return payload


# ─────────────────────────────────────────────────────────────────
# Convenience wrapper for existing callers (e.g. bootstrap_models.py)
# ─────────────────────────────────────────────────────────────────
def train_yield_model(output_path: str = "models/yield_xgboost.pkl"):
    """Backward-compatible entry point for initializing the yield model."""
    d, gr = make_demo_data(n=5000, base_yield=24.0, seed=42)
    return train(d, gr, unit="t_per_acre", data_source="synthetic_demo", out_path=output_path, seed=42)


# ─────────────────────────────────────────────────────────────────
# CLI Entrypoint
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train Fasal Suraksha XGBoost Yield Model")
    mode = ap.add_mutually_exclusive_group(required=False)
    mode.add_argument("--csv", help="Path to real field data CSV")
    mode.add_argument("--demo", action="store_true", help="Generate synthetic agronomic dataset")
    ap.add_argument("--unit", choices=["t_per_ha", "t_per_acre"], default="t_per_acre")
    ap.add_argument("--target", default="actual_yield")
    ap.add_argument("--group_col", default="farm_id")
    ap.add_argument("--demo_base_yield", type=float, default=24.0)
    ap.add_argument("--output", default="models/yield_xgboost.pkl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.csv:
        data_df, group_arr = load_real_csv(args.csv, args.target, args.group_col)
        source = "real_csv"
    else:
        data_df, group_arr = make_demo_data(base_yield=args.demo_base_yield, seed=args.seed)
        source = "synthetic_demo"

    train(data_df, group_arr, unit=args.unit, data_source=source, out_path=args.output, seed=args.seed)
