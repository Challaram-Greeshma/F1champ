from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f1champ.constants import LOGISTIC_L2


@dataclass
class RetirementModel:
    intercept: float
    coef: float
    rate_mean: float
    rate_std: float

    def predict_proba(self, rates: np.ndarray) -> np.ndarray:
        values = np.asarray(rates, dtype=float)
        values = np.where(np.isnan(values), self.rate_mean, values)
        scale = self.rate_std if self.rate_std > 1e-8 else 1.0
        eta = np.clip(self.intercept + self.coef * ((values - self.rate_mean) / scale), -20, 20)
        return 1.0 / (1.0 + np.exp(-eta))


def incident_base_rate(features: pd.DataFrame, outcomes: pd.DataFrame) -> float:
    import pandas as pd

    merged = _starters(features, outcomes)
    if merged.empty:
        return 0.05
    return float(merged["retired_incident"].astype(bool).mean())


def fit_retirement(features: pd.DataFrame, outcomes: pd.DataFrame, l2: float = LOGISTIC_L2) -> RetirementModel:
    import pandas as pd

    merged = _starters(features, outcomes)
    rates = pd.to_numeric(merged["constructor_mech_rate"], errors="coerce")
    observed = rates.dropna()
    rate_mean = float(observed.mean()) if not observed.empty else 0.1
    rate_std = float(observed.std(ddof=0)) if len(observed) > 1 else 1.0
    if rate_std < 1e-8:
        rate_std = 1.0
    if merged.empty:
        return RetirementModel(intercept=_logit(0.05), coef=0.0, rate_mean=rate_mean, rate_std=rate_std)

    y = merged["retired_mechanical"].astype(float).to_numpy()
    x = ((rates.fillna(rate_mean).to_numpy(dtype=float) - rate_mean) / rate_std)
    if y.min() == y.max():
        return RetirementModel(intercept=_logit(float(y.mean())), coef=0.0, rate_mean=rate_mean, rate_std=rate_std)

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        eta = np.clip(theta[0] + theta[1] * x, -20, 20)
        nll = float(np.mean(_softplus(eta) - y * eta) + 0.5 * l2 * theta[1] ** 2)
        residual = _sigmoid(eta) - y
        grad = np.array(
            [
                float(np.mean(residual)),
                float(np.mean(residual * x) + l2 * theta[1]),
            ]
        )
        return nll, grad

    from scipy.optimize import minimize

    result = minimize(objective, np.zeros(2), jac=True, method="L-BFGS-B")
    theta = result.x
    return RetirementModel(
        intercept=float(theta[0]),
        coef=float(theta[1]),
        rate_mean=rate_mean,
        rate_std=rate_std,
    )


def _starters(features, outcomes):
    merged = features.merge(outcomes, on=["season", "round", "driver_id"], how="inner")
    return merged[merged["status_category"] != "dns"].copy()


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-eta))


def _softplus(eta: np.ndarray) -> np.ndarray:
    return np.maximum(eta, 0) + np.log1p(np.exp(-np.abs(eta)))


def _logit(probability: float) -> float:
    p = min(max(probability, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1 - p)))
