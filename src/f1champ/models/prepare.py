from __future__ import annotations

import pandas as pd


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Grid and Elo transforms that use only the current entry list."""
    out = frame.copy()
    grid = pd.to_numeric(out["grid_position"], errors="coerce")
    valid = grid.where(grid >= 1)
    back = valid.groupby([out["season"], out["round"]], sort=False).transform("max")
    out["grid_model"] = valid.fillna(back + 1).fillna(20.0)
    driver = pd.to_numeric(out["driver_elo"], errors="coerce").fillna(1500.0)
    constructor = pd.to_numeric(out["constructor_elo"], errors="coerce").fillna(1500.0)
    out["elo_sum"] = driver + constructor
    return out
