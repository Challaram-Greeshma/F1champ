from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from f1champ.constants import PL_L2
from f1champ.models.prepare import add_derived_features
from f1champ.schema import assert_model_columns


@dataclass
class PlackettLuceModel:
    feature_names: list[str]
    beta: np.ndarray
    mean: np.ndarray
    std: np.ndarray

    def skills(self, frame: pd.DataFrame) -> np.ndarray:
        design = design_matrix(frame, self.feature_names, self.mean, self.std)
        return design @ self.beta


def fit_plackett_luce(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    feature_names: list[str],
    l2: float = PL_L2,
) -> PlackettLuceModel:
    assert_model_columns(feature_names)
    merged = features.merge(outcomes, on=["season", "round", "driver_id"], how="inner")
    prepared = add_derived_features(merged)
    starters = prepared[prepared["status_category"] != "dns"]
    mean, std = _moments(starters if not starters.empty else prepared, feature_names)
    races: list[np.ndarray] = []
    classified = prepared[prepared["classified"].astype(bool)]
    for (_, _), group in classified.groupby(["season", "round"], sort=False):
        ordered = group.sort_values(["finish_position", "driver_id"])
        if len(ordered) < 2:
            continue
        races.append(design_matrix(ordered, feature_names, mean, std))

    width = len(feature_names)
    if not races:
        return PlackettLuceModel(list(feature_names), np.zeros(width), mean, std)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        nll = 0.5 * l2 * float(np.dot(beta, beta))
        grad = l2 * beta.astype(float)
        for design in races:
            skills = design @ beta
            race_nll, skill_grad = race_nll_and_grad(skills)
            nll += race_nll
            grad = grad + design.T @ skill_grad
        return nll, grad

    result = minimize(objective, np.zeros(width), jac=True, method="L-BFGS-B", options={"maxiter": 200})
    return PlackettLuceModel(list(feature_names), np.asarray(result.x, dtype=float), mean, std)


def design_matrix(
    frame: pd.DataFrame,
    feature_names: list[str],
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    columns = []
    for index, name in enumerate(feature_names):
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float)
        filled = np.where(np.isnan(values), mean[index], values)
        scale = std[index] if std[index] > 1e-8 else 1.0
        columns.append((filled - mean[index]) / scale)
    if not columns:
        return np.zeros((len(frame), 0))
    return np.column_stack(columns)


def race_nll_and_grad(skills_in_order: np.ndarray) -> tuple[float, np.ndarray]:
    """Negative Plackett-Luce log-likelihood for one finishing order, best first."""
    skills = np.asarray(skills_in_order, dtype=float)
    n = len(skills)
    grad = np.zeros(n)
    if n < 2:
        return 0.0, grad
    nll = 0.0
    for start in range(n - 1):
        remaining = skills[start:]
        peak = float(np.max(remaining))
        shifted = np.exp(remaining - peak)
        denom = float(np.sum(shifted))
        nll += -(skills[start] - (peak + np.log(denom)))
        probs = shifted / denom
        grad[start:] += probs
        grad[start] -= 1.0
    return nll, grad


def _moments(frame: pd.DataFrame, feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    means = []
    stds = []
    for name in feature_names:
        values = pd.to_numeric(frame[name], errors="coerce") if name in frame.columns else pd.Series(dtype=float)
        observed = values.dropna()
        if observed.empty:
            means.append(0.0)
            stds.append(1.0)
            continue
        deviation = float(observed.std(ddof=0))
        means.append(float(observed.mean()))
        stds.append(deviation if deviation > 1e-8 else 1.0)
    return np.asarray(means, dtype=float), np.asarray(stds, dtype=float)
