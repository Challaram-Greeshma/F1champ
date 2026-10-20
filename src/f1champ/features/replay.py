from __future__ import annotations

import pandas as pd

from f1champ.constants import CUTOFF
from f1champ.features.ratings import DriverResult, RatingState
from f1champ.schema import FEATURE_COLUMNS, OUTCOME_COLUMNS


def build_feature_tables(
    races: pd.DataFrame,
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
    sprints: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One forward pass. Features for a race are written before that race updates ratings."""
    if races.empty:
        return _empty_features(), _empty_outcomes()

    calendar = races.copy()
    calendar["season"] = calendar["season"].astype(int)
    calendar["round"] = calendar["round"].astype(int)
    calendar["race_date"] = pd.to_datetime(calendar["race_date"], errors="coerce")
    calendar = calendar.dropna(subset=["race_date"]).sort_values(["race_date", "season", "round"])

    results = results.copy()
    results["season"] = results["season"].astype(int)
    results["round"] = results["round"].astype(int)
    qualifying = qualifying.copy() if qualifying is not None and not qualifying.empty else pd.DataFrame()
    if not qualifying.empty:
        qualifying["season"] = qualifying["season"].astype(int)
        qualifying["round"] = qualifying["round"].astype(int)
    sprints = sprints.copy() if sprints is not None and not sprints.empty else pd.DataFrame()
    sprint_keys: set[tuple[int, int]] = set()
    if not sprints.empty:
        sprint_keys = set(zip(sprints["season"].astype(int), sprints["round"].astype(int)))

    state = RatingState()
    feature_rows: list[dict] = []
    outcome_rows: list[dict] = []
    previous_date = None

    for race in calendar.itertuples(index=False):
        season = int(race.season)
        rnd = int(race.round)
        race_date = pd.Timestamp(race.race_date)
        circuit_id = race.circuit_id
        race_results = results[(results["season"] == season) & (results["round"] == rnd)]
        race_quali = (
            qualifying[(qualifying["season"] == season) & (qualifying["round"] == rnd)]
            if not qualifying.empty
            else pd.DataFrame()
        )
        if race_results.empty and race_quali.empty:
            continue

        days_since = None if previous_date is None else float((race_date - previous_date).days)
        previous_date = race_date
        state.start_season_if_needed(season)
        quali_by_driver = _qualifying_lookup(race_quali)
        entries = _entry_list(race_results, race_quali)
        sprint_held = 1 if (season, rnd) in sprint_keys else 0

        for entry in entries:
            ratings = state.pre_race_features(entry["driver_id"], entry["constructor_id"], circuit_id)
            quali_position, quali_gap, has_time = quali_by_driver.get(
                entry["driver_id"], (None, None, False)
            )
            grid = entry["grid"]
            has_grid = grid is not None and grid >= 1
            feature_rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "race_date": race_date.date().isoformat(),
                    "circuit_id": circuit_id,
                    "driver_id": entry["driver_id"],
                    "constructor_id": entry["constructor_id"],
                    "cutoff": CUTOFF,
                    "grid_position": grid if has_grid else None,
                    "quali_position": quali_position,
                    "quali_gap_s": quali_gap,
                    "driver_elo": ratings["driver_elo"],
                    "constructor_elo": ratings["constructor_elo"],
                    "teammate_residual": ratings["teammate_residual"],
                    "constructor_mech_rate": ratings["constructor_mech_rate"],
                    "driver_incident_rate": ratings["driver_incident_rate"],
                    "circuit_residual": ratings["circuit_residual"],
                    "days_since_last_race": days_since,
                    "sprint_held": sprint_held,
                    "has_grid": bool(has_grid),
                    "has_qualifying_time": bool(has_time),
                    "has_sprint": bool(sprint_held),
                    "has_lap_data": False,
                }
            )

        if race_results.empty:
            continue
        outcome_rows.extend(_outcomes(race_results, season, rnd))
        state.update_from_results(_driver_results(race_results), circuit_id)

    features = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
    outcomes = pd.DataFrame(outcome_rows, columns=OUTCOME_COLUMNS)
    return features, outcomes


def _qualifying_lookup(race_quali: pd.DataFrame) -> dict[str, tuple]:
    if race_quali.empty:
        return {}
    representative = pd.to_numeric(race_quali["representative_s"], errors="coerce")
    pole = representative.min()
    lookup: dict[str, tuple] = {}
    for row, rep in zip(race_quali.itertuples(index=False), representative):
        gap = None if pd.isna(rep) or pd.isna(pole) else float(rep - pole)
        position = None if pd.isna(row.quali_position) else int(row.quali_position)
        lookup[row.driver_id] = (position, gap, bool(pd.notna(rep)))
    return lookup


def _entry_list(race_results: pd.DataFrame, race_quali: pd.DataFrame) -> list[dict]:
    entries: dict[str, dict] = {}
    if not race_quali.empty:
        for row in race_quali.itertuples(index=False):
            entries[row.driver_id] = {
                "driver_id": row.driver_id,
                "constructor_id": row.constructor_id,
                "grid": None,
            }
    if not race_results.empty:
        for row in race_results.itertuples(index=False):
            grid = row.grid if pd.notna(row.grid) else None
            grid = int(grid) if grid is not None else None
            current = entries.get(row.driver_id, {"driver_id": row.driver_id, "constructor_id": None, "grid": None})
            current["constructor_id"] = row.constructor_id or current["constructor_id"]
            current["grid"] = grid
            entries[row.driver_id] = current
    rows = []
    for key in sorted(entries):
        entry = entries[key]
        if not entry["constructor_id"]:
            entry["constructor_id"] = "unknown"
        rows.append(entry)
    return rows


def _driver_results(race_results: pd.DataFrame) -> list[DriverResult]:
    rows: list[DriverResult] = []
    for row in race_results.itertuples(index=False):
        position = int(row.finish_position) if pd.notna(row.finish_position) else None
        rows.append(
            DriverResult(
                driver_id=row.driver_id,
                constructor_id=row.constructor_id,
                finish_position=position,
                status_category=row.status_category,
            )
        )
    return rows


def _outcomes(race_results: pd.DataFrame, season: int, rnd: int) -> list[dict]:
    times = pd.to_numeric(race_results["race_time_ms"], errors="coerce")
    classified = race_results["status_category"].eq("finished")
    winner = race_results[classified & race_results["finish_position"].eq(1)]
    if not winner.empty and pd.notna(times.loc[winner.index[0]]):
        winner_ms = float(times.loc[winner.index[0]])
    elif classified.any() and times[classified].notna().any():
        winner_ms = float(times[classified].min())
    else:
        winner_ms = None

    rows = []
    for index, row in race_results.iterrows():
        category = row["status_category"]
        position = int(row["finish_position"]) if pd.notna(row["finish_position"]) else None
        is_classified = category == "finished"
        gap = None
        if is_classified and winner_ms is not None and pd.notna(times.loc[index]):
            gap = (float(times.loc[index]) - winner_ms) / 1000.0
        rows.append(
            {
                "season": season,
                "round": rnd,
                "driver_id": row["driver_id"],
                "finish_position": position,
                "classified": bool(is_classified),
                "retired_mechanical": category == "mechanical",
                "retired_incident": category == "incident",
                "same_lap_gap_s": gap,
                "won": bool(is_classified and position == 1),
                "podium": bool(is_classified and position is not None and position <= 3),
                "status_category": category,
                "points": None if pd.isna(row["points"]) else float(row["points"]),
                "laps_completed": None if pd.isna(row["laps"]) else int(row["laps"]),
            }
        )
    return rows


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_COLUMNS)


def _empty_outcomes() -> pd.DataFrame:
    return pd.DataFrame(columns=OUTCOME_COLUMNS)
