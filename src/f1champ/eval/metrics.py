from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def ranking_metrics(frame: pd.DataFrame) -> dict:
    """Position metrics use classified finishers. Win metrics use the whole field."""
    if frame.empty:
        return _empty_ranking()

    absolute_errors: list[np.ndarray] = []
    correlations: list[float] = []
    log_losses: list[float] = []
    briers: list[float] = []

    for (_, _), race in frame.groupby(["season", "round"], sort=False):
        classified = race["classified"].fillna(False).astype(bool) & race["finish_position"].notna()
        if classified.any() and race.loc[classified, "expected_position"].notna().all():
            error = (
                race.loc[classified, "expected_position"].to_numpy(dtype=float)
                - race.loc[classified, "finish_position"].to_numpy(dtype=float)
            )
            absolute_errors.append(np.abs(error))
        if int(classified.sum()) >= 2:
            left = race.loc[classified, "expected_position"].to_numpy(dtype=float)
            right = race.loc[classified, "finish_position"].to_numpy(dtype=float)
            if np.nanstd(left) > 0 and np.nanstd(right) > 0:
                corr = float(spearmanr(left, right).statistic)
                if math.isfinite(corr):
                    correlations.append(corr)

        won = race["won"].fillna(False).astype(bool)
        if int(won.sum()) != 1:
            continue
        probs = np.clip(race["win_probability"].to_numpy(dtype=float), 1e-15, 1.0)
        total = float(probs.sum())
        if total <= 0:
            continue
        probs = probs / total
        log_losses.append(-math.log(float(probs[won.to_numpy()][0])))
        actual = won.astype(float).to_numpy()
        briers.append(float(np.sum((actual - probs) ** 2)))

    return {
        "position_mae": _flat_mean(absolute_errors),
        "spearman": _mean(correlations),
        "win_log_loss": _mean(log_losses),
        "win_brier": _mean(briers),
        "n_races": int(frame.groupby(["season", "round"]).ngroups),
        "n_classified": int(
            (frame["classified"].fillna(False).astype(bool) & frame["finish_position"].notna()).sum()
        ),
    }


def retirement_metrics(frame: pd.DataFrame, n_bins: int = 10) -> dict:
    if frame.empty or "status_category" not in frame.columns:
        return {"brier": None, "calibration": [], "n_starters": 0}
    starters = frame[frame["status_category"].notna() & frame["status_category"].ne("dns")]
    if starters.empty:
        return {"brier": None, "calibration": [], "n_starters": 0}
    actual = starters["retired_mechanical"].fillna(False).astype(float).to_numpy()
    predicted = np.clip(starters["mechanical_retirement_probability"].to_numpy(dtype=float), 0.0, 1.0)
    return {
        "brier": float(np.mean((actual - predicted) ** 2)),
        "calibration": calibration_table(actual, predicted, n_bins=n_bins),
        "n_starters": int(len(starters)),
    }


def calibration_table(actual: np.ndarray, predicted: np.ndarray, n_bins: int = 10) -> list[dict]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for index in range(n_bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == n_bins - 1:
            mask = (predicted >= lower) & (predicted <= upper)
        else:
            mask = (predicted >= lower) & (predicted < upper)
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_predicted": float(predicted[mask].mean()),
                "observed_rate": float(actual[mask].mean()),
            }
        )
    return rows


def keep_feature(stage1_log_loss: dict[int, float], candidate_log_loss: dict[int, float]) -> dict:
    """Keep a candidate only when mean season log loss improves in a majority of seasons."""
    seasons = sorted(set(stage1_log_loss) & set(candidate_log_loss))
    improvements = {}
    for season in seasons:
        base = stage1_log_loss[season]
        challenger = candidate_log_loss[season]
        if base is None or challenger is None:
            continue
        if not math.isfinite(base) or not math.isfinite(challenger):
            continue
        improvements[str(season)] = float(base - challenger)
    values = list(improvements.values())
    mean_improvement = float(np.mean(values)) if values else 0.0
    seasons_improved = int(sum(value > 0 for value in values))
    kept = bool(values) and mean_improvement > 0 and seasons_improved > len(values) / 2
    return {
        "log_loss_improvement_by_season": improvements,
        "mean_log_loss_improvement": mean_improvement,
        "seasons_improved": seasons_improved,
        "seasons_compared": len(values),
        "kept": kept,
    }


def _empty_ranking() -> dict:
    return {
        "position_mae": None,
        "spearman": None,
        "win_log_loss": None,
        "win_brier": None,
        "n_races": 0,
        "n_classified": 0,
    }


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.mean(values))


def _flat_mean(chunks: list[np.ndarray]) -> float | None:
    if not chunks:
        return None
    return float(np.concatenate(chunks).mean())
