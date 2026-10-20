from __future__ import annotations

import numpy as np
import pandas as pd

from f1champ.models.prepare import add_derived_features
from f1champ.models.retirement import RetirementModel
from f1champ.models.simulate import simulate_race
from f1champ.schema import PREDICTION_FEATURE_FIELDS


def forecast_race(
    model,
    retirement: RetirementModel,
    incident_rate: float,
    race_features: pd.DataFrame,
    n_sim: int,
    rng: np.random.Generator,
    *,
    zero_retirement: bool = False,
) -> pd.DataFrame:
    """Return one row per driver. Drivers are sorted by id so simulations are reproducible."""
    frame = add_derived_features(race_features).sort_values(["driver_id", "constructor_id"]).reset_index(drop=True)
    skills = model.skills(frame)
    if zero_retirement:
        p_mech = np.zeros(len(frame))
        incident = 0.0
    else:
        rates = pd.to_numeric(frame["constructor_mech_rate"], errors="coerce").to_numpy(dtype=float)
        p_mech = retirement.predict_proba(rates)
        incident = incident_rate
    simulated = simulate_race(skills, p_mech, incident, n_sim, rng)
    out = frame.copy()
    out["expected_position"] = simulated.expected_position
    out["win_probability"] = simulated.win_probability
    out["podium_probability"] = simulated.podium_probability
    out["mechanical_retirement_probability"] = simulated.mechanical_retirement_probability
    out["retirement_probability"] = simulated.retirement_probability
    out["position_distribution"] = list(simulated.position_distribution)
    return out


def feature_snapshot(row: pd.Series) -> dict:
    snapshot = {}
    for name in PREDICTION_FEATURE_FIELDS:
        if name not in row.index:
            continue
        snapshot[name] = _clean(row[name])
    return snapshot


def _clean(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    return value
