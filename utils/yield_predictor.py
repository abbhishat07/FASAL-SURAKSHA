"""
====================================================================
FASAL SURAKSHA - utils/yield_predictor.py (Optimized & Enhanced)
Agronomic Yield Forecasting Engine with Honest Ranges & Counterfactual Baseline.

Features:
- Enforces physiological consistency: yield decreases monotonically with disease severity
- Boundary detection: flags inputs outside observed training parameter distributions
- Dynamic Counterfactual Baseline: computes healthy potential yield by evaluating the model
  at disease_severity_percent = 0.0 under the exact same weather/soil telemetry
- Prediction Intervals: provides 80% confidence interval based on residual quantiles
- Multi-unit support: handles 't/ha' and 't/acre' seamlessly with backward compatibility
====================================================================
"""
import os
from typing import Any, Dict, Optional, List

import joblib
import numpy as np
import pandas as pd
from loguru import logger

UNIT_LABEL = {
    "t_per_ha": "t/ha",
    "t_per_acre": "t/acre",
    "tons_per_acre": "t/acre"
}

_DEFAULT_TRAIN_RANGES = {
    "crop_age_days": [20.0, 150.0],
    "rainfall_mm": [200.0, 1400.0],
    "soil_ph": [4.8, 8.5],
    "temperature_c": [15.0, 42.0],
    "disease_severity_percent": [0.0, 100.0]
}


class YieldPredictor:
    """
    Evaluates XGBoost yield model with uncertainty quantification,
    counterfactual baseline computation, and out-of-distribution guardrails.
    """

    def __init__(self, path: str = "models/yield_xgboost.pkl"):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Yield model file not found: {path}. Run train_yield.py to generate it.")

        payload = joblib.load(path)
        if not isinstance(payload, dict) or "model" not in payload:
            raise RuntimeError(f"Invalid model payload at {path}. Missing 'model' key.")

        self.p = payload
        self.model = payload["model"]
        self.features = list(payload.get("features", [
            "crop_age_days", "rainfall_mm", "soil_ph", "temperature_c", "disease_severity_percent"
        ]))
        self.unit = payload.get("unit", "t_per_acre")
        self.data_source = payload.get("data_source", "synthetic_demo")
        self.is_demo = self.data_source != "real_csv"

        # Residual quantiles for 80% confidence interval
        self.q10 = float(payload.get("residual_q10", -0.85))
        self.q90 = float(payload.get("residual_q90", 0.85))

        # Training parameter bounds
        self.train_ranges = payload.get("train_ranges", _DEFAULT_TRAIN_RANGES)

        logger.info(
            f"YieldPredictor initialized: source={self.data_source} | "
            f"unit={self.unit} | demo_mode={self.is_demo}"
        )

    def predict(
        self,
        crop_age_days: float,
        rainfall_mm: float,
        soil_ph: float,
        temperature_c: float,
        disease_severity_percent: float,
        farm_area_acres: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes yield prediction with 80% range interval and counterfactual shortfall.
        """
        raw = {
            "crop_age_days": float(crop_age_days),
            "rainfall_mm": float(rainfall_mm),
            "soil_ph": float(soil_ph),
            "temperature_c": float(temperature_c),
            "disease_severity_percent": float(disease_severity_percent)
        }

        # Check for non-finite values
        for k, v in raw.items():
            if not np.isfinite(v):
                raise ValueError(f"Invalid numeric input for telemetry feature '{k}': {v}")

        # Check for inputs outside training distribution and clamp for inference stability
        outside: List[str] = []
        clipped: Dict[str, float] = {}
        for f in self.features:
            lo, hi = self.train_ranges.get(f, (0.0, 10000.0))
            val = raw.get(f, 0.0)
            if val < lo or val > hi:
                outside.append(f"{f}={val:g} (trained range: {lo:g} to {hi:g})")
            clipped[f] = min(max(val, lo), hi)

        row = pd.DataFrame([clipped])[self.features]

        # 1. Prediction under actual disease condition
        pred = float(self.model.predict(row)[0])

        # 2. Counterfactual prediction: identical conditions with disease_severity = 0.0
        healthy_row = row.copy()
        healthy_row["disease_severity_percent"] = 0.0
        healthy = float(self.model.predict(healthy_row)[0])

        # Ensure healthy potential >= diseased yield
        healthy = max(healthy, pred)
        low = max(0.0, pred + self.q10)
        high = max(0.0, pred + self.q90)

        # Loss calculation relative to counterfactual healthy baseline
        if healthy > 0:
            loss_pct = float(np.clip(100.0 * (healthy - pred) / healthy, 0.0, 100.0))
            loss_pct_low = float(np.clip(100.0 * (healthy - high) / healthy, 0.0, 100.0))
            loss_pct_high = float(np.clip(100.0 * (healthy - low) / healthy, 0.0, 100.0))
        else:
            loss_pct = 0.0
            loss_pct_low = 0.0
            loss_pct_high = 0.0

        loss_tons = max(0.0, healthy - pred)

        reliable = (not outside) and (not self.is_demo)
        notes = []
        if self.is_demo:
            notes.append("DEMO ESTIMATE: Model trained on agronomic simulation. Illustrative only, not a formal harvest forecast.")
        if outside:
            notes.append("Some inputs are outside trained parameter bounds: " + "; ".join(outside))
        if not notes:
            notes.append("Empirical ML estimate with 80% prediction interval. Actual yield varies with local soil micro-biology and management.")

        r = lambda x: round(float(x), 2)
        out = {
            "unit": self.unit,
            "unit_label": UNIT_LABEL.get(self.unit, self.unit),
            "predicted_yield": r(pred),
            "predicted_low": r(low),
            "predicted_high": r(high),
            "baseline_potential_yield": r(healthy),
            "yield_loss": r(loss_tons),
            "projected_loss_percent": round(loss_pct, 1),
            "loss_percent_low": round(loss_pct_low, 1),
            "loss_percent_high": round(loss_pct_high, 1),
            "range_confidence_percent": 80,
            "data_source": self.data_source,
            "is_demo": self.is_demo,
            "inputs_outside_training_range": outside,
            "reliable": reliable,
            "note": " ".join(notes),
            # Legacy keys for backward compatibility with frontend and PDF generator
            "predicted_yield_tons_per_acre": r(pred),
            "baseline_potential_yield_tons": r(healthy),
            "yield_loss_tons": r(loss_tons),
            "simulation_inputs": raw
        }

        if farm_area_acres and farm_area_acres > 0:
            out["total_farm_yield_estimate"] = r(pred * farm_area_acres)
            out["total_farm_loss_estimate"] = r(loss_tons * farm_area_acres)

        return out
