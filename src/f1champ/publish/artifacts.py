from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from f1champ.constants import CUTOFF, LIMITATIONS, MODEL_VERSION, N_SIM
from f1champ.models.forecast import feature_snapshot
from f1champ.schema import STAGE1_FEATURES


def race_has_qualifying_info(frame: pd.DataFrame) -> bool:
    if frame is None or frame.empty:
        return False
    grid = pd.to_numeric(frame["grid_position"], errors="coerce")
    if bool(grid.ge(1).any()):
        return True
    if "has_grid" in frame.columns and bool(frame["has_grid"].fillna(False).astype(bool).any()):
        return True
    if "has_qualifying_time" in frame.columns and bool(
        frame["has_qualifying_time"].fillna(False).astype(bool).any()
    ):
        return True
    if "quali_position" in frame.columns and bool(frame["quali_position"].notna().any()):
        return True
    return False


def post_qualifying_message(season: int, rnd: int) -> str:
    return (
        f"This model is post-qualifying. Season {season} round {rnd} "
        "has no grid or qualifying result yet."
    )


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2), encoding="utf-8")


def write_race_prediction(
    directory: Path,
    race_features: pd.DataFrame,
    n_sim: int = N_SIM,
) -> Path:
    season = int(race_features["season"].iloc[0])
    rnd = int(race_features["round"].iloc[0])
    race_date = race_features["race_date"].iloc[0]
    circuit_id = race_features["circuit_id"].iloc[0]
    drivers = []
    ordered = race_features.sort_values(["expected_position", "driver_id"])
    for _, row in ordered.iterrows():
        distribution = row["position_distribution"]
        drivers.append(
            {
                "driver_id": row["driver_id"],
                "constructor_id": row["constructor_id"],
                "features": feature_snapshot(row),
                "expected_position": _round_or_none(row["expected_position"], 3),
                "win_probability": _round_or_none(row["win_probability"], 6),
                "podium_probability": _round_or_none(row["podium_probability"], 6),
                "retirement_probability": _round_or_none(row["retirement_probability"], 6),
                "mechanical_retirement_probability": _round_or_none(
                    row["mechanical_retirement_probability"], 6
                ),
                "position_distribution": {
                    str(place): round(float(probability), 6)
                    for place, probability in enumerate(distribution, start=1)
                },
            }
        )
    payload = {
        "model_version": MODEL_VERSION,
        "cutoff": CUTOFF,
        "season": season,
        "round": rnd,
        "race_date": None if pd.isna(race_date) else str(race_date),
        "circuit_id": None if pd.isna(circuit_id) else str(circuit_id),
        "n_sim": int(n_sim),
        "features_used": list(STAGE1_FEATURES),
        "expected_position_is": "conditional_on_classified",
        "limitations": list(LIMITATIONS),
        "drivers": drivers,
    }
    path = directory / f"{season}_{rnd:02d}.json"
    write_json(path, payload)
    return path


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _round_or_none(value, digits: int):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)
