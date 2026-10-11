"""
====================================================================
FASAL SURAKSHA - train_yield.py
Agronomic Yield Prediction Model (Tabular ML).
Trains an XGBoost Regressor on field telemetry:
  - crop_age_days (e.g. 20 - 150 days)
  - rainfall_mm (e.g. 200 - 1400 mm)
  - soil_ph (e.g. 5.0 - 8.5)
  - temperature_c (e.g. 15.0 - 42.0 C)
  - disease_severity_percent (0.0 - 100.0%)
Target:
  - actual_yield_tons (tons / hectare or acre)
Exports:
  models/yield_xgboost.pkl
====================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor


def generate_agronomic_yield_dataset(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """
    Synthesizes agronomically grounded crop harvest data.
    Models physiological responses:
      - Optimal pH between 6.0 and 7.2
      - Optimal rainfall curve with drought and waterlogging penalties
      - Age maturity saturation
      - Pathological necrosis penalizing photosynthetic area exponentially
    """
    np.random.seed(random_seed)

    crop_age_days = np.random.uniform(30.0, 130.0, n_samples)
    rainfall_mm   = np.random.uniform(250.0, 1200.0, n_samples)
    soil_ph       = np.random.uniform(5.2, 8.2, n_samples)
    temperature_c = np.random.uniform(18.0, 38.0, n_samples)
    disease_severity_percent = np.random.uniform(0.0, 95.0, n_samples)

    # Standard healthy yield potential (Tomato/Vegetables benchmark ~ 24.0 Tons/Acre)
    base_potential_tons = 24.0

    # 1. Soil pH factor (peak at pH 6.5)
    ph_factor = np.exp(-0.5 * ((soil_ph - 6.5) / 0.8) ** 2)

    # 2. Rainfall factor (optimal 650 mm)
    rain_factor = np.exp(-0.5 * ((rainfall_mm - 650.0) / 280.0) ** 2)

    # 3. Temperature factor (optimal 26 C)
    temp_factor = np.exp(-0.5 * ((temperature_c - 26.0) / 6.0) ** 2)

    # 4. Crop age growth completion curve (sigmoidal approaching maturity ~ 90-110 days)
    maturity_factor = 1.0 / (1.0 + np.exp(-(crop_age_days - 60.0) / 12.0))

    # 5. Disease severity damage multiplier (non-linear loss)
    # 0% severity = 1.0 (no damage); 100% severity = down to 0.15 (85% destruction)
    disease_damage_factor = 1.0 - (0.85 * (disease_severity_percent / 100.0) ** 1.3)

    # Theoretical actual yield
    actual_yield_tons = (
        base_potential_tons
        * (0.3 + 0.7 * maturity_factor)
        * (0.6 + 0.4 * ph_factor)
        * (0.6 + 0.4 * rain_factor)
        * (0.7 + 0.3 * temp_factor)
        * disease_damage_factor
    )

    # Add real-world stochastic farm noise (wind, pests, soil variance)
    noise = np.random.normal(0, 0.4, n_samples)
    actual_yield_tons = np.clip(actual_yield_tons + noise, 0.5, 30.0)

    df = pd.DataFrame({
        "crop_age_days": crop_age_days.round(1),
        "rainfall_mm": rainfall_mm.round(1),
        "soil_ph": soil_ph.round(2),
        "temperature_c": temperature_c.round(1),
        "disease_severity_percent": disease_severity_percent.round(1),
        "actual_yield_tons": actual_yield_tons.round(2)
    })
    return df


def train_yield_model(output_path: str = "models/yield_xgboost.pkl"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    logger.info("Generating agronomic telemetry dataset for yield forecasting...")

    df = generate_agronomic_yield_dataset(n_samples=6000)

    feature_cols = [
        "crop_age_days",
        "rainfall_mm",
        "soil_ph",
        "temperature_c",
        "disease_severity_percent"
    ]
    target_col = "actual_yield_tons"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    logger.info(f"Training XGBoost Regressor on {len(X_train)} samples...")
    model = XGBRegressor(
        n_estimators=250,
        learning_rate=0.04,
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Evaluation
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    logger.info(f"Yield Model Evaluation:")
    logger.info(f"  R-Squared (R²): {r2:.4f}")
    logger.info(f"  RMSE:           {rmse:.3f} Tons")
    logger.info(f"  MAE:            {mae:.3f} Tons")

    # Save artifact
    payload = {
        "model": model,
        "features": feature_cols,
        "metrics": {"r2": r2, "rmse": rmse, "mae": mae},
        "baseline_potential_yield_tons": 24.0
    }
    joblib.dump(payload, output_path)
    logger.info(f"Trained XGBoost yield model dumped successfully to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train XGBoost Yield Regressor")
    parser.add_argument("--output", type=str, default="models/yield_xgboost.pkl")
    args = parser.parse_args()

    train_yield_model(output_path=args.output)
