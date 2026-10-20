from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from f1champ.models.prepare import add_derived_features
from f1champ.models.ranker import PlackettLuceModel, fit_plackett_luce


@dataclass
class GridBaseline:
    """Historical win rate by starting slot, smoothed, then renormalized per race."""

    win_rate_by_grid: dict[int, float]
    default_rate: float


def fit_grid_baseline(features, outcomes) -> GridBaseline:
    merged = features.merge(outcomes, on=["season", "round", "driver_id"], how="inner")
    starters = merged[merged["status_category"] != "dns"]
    grid = starters["grid_position"]
    valid = starters[pd_to_numeric_ge(grid)].copy()
    rates: dict[int, float] = {}
    if not valid.empty:
        valid["grid_slot"] = valid["grid_position"].astype(int)
        for slot, group in valid.groupby("grid_slot"):
            wins = int(group["won"].astype(bool).sum())
            rates[int(slot)] = (wins + 0.5) / (len(group) + 1.0)
    default = min(rates.values()) if rates else 0.01
    return GridBaseline(rates, default)


def grid_expected_positions(frame) -> np.ndarray:
    prepared = add_derived_features(frame)
    return prepared["grid_model"].rank(method="average", ascending=True).to_numpy(dtype=float)


def grid_win_probabilities(frame, baseline: GridBaseline) -> np.ndarray:
    prepared = add_derived_features(frame)
    raw = []
    for value in prepared["grid_position"]:
        if value is None or (isinstance(value, float) and np.isnan(value)) or pd_isna(value):
            raw.append(baseline.default_rate)
            continue
        try:
            slot = int(value)
        except (TypeError, ValueError):
            raw.append(baseline.default_rate)
            continue
        if slot < 1:
            raw.append(baseline.default_rate)
        else:
            raw.append(baseline.win_rate_by_grid.get(slot, baseline.default_rate))
    probs = np.asarray(raw, dtype=float)
    total = float(probs.sum())
    if total <= 0:
        return np.full(len(probs), 1.0 / max(len(probs), 1))
    return probs / total


def fit_ratings_baseline(features, outcomes) -> PlackettLuceModel:
    return fit_plackett_luce(features, outcomes, ["elo_sum"])


def ratings_expected_positions(frame) -> np.ndarray:
    prepared = add_derived_features(frame)
    return prepared["elo_sum"].rank(method="average", ascending=False).to_numpy(dtype=float)


def pd_isna(value) -> bool:
    try:
        return bool(value != value)
    except Exception:
        return False


def pd_to_numeric_ge(series) -> "object":
    import pandas as pd

    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.ge(1).fillna(False)
