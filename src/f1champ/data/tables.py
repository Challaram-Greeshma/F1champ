from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from f1champ.constants import START_SEASON
from f1champ.data.jolpica import (
    JolpicaClient,
    JolpicaError,
    parse_circuits,
    parse_constructors,
    parse_drivers,
    parse_qualifying,
    parse_races,
    parse_results,
    parse_seasons,
    parse_sprints,
    parse_status_table,
)
from f1champ.data.status import classify_status
from f1champ.paths import DataPaths

logger = logging.getLogger(__name__)

RESULT_COLUMNS = [
    "season",
    "round",
    "race_date",
    "circuit_id",
    "driver_id",
    "constructor_id",
    "car_number",
    "grid",
    "finish_position",
    "position_text",
    "points",
    "laps",
    "status",
    "status_category",
    "race_time_ms",
    "has_grid",
    "has_race_time",
    "has_qualifying_time",
    "has_sprint",
    "has_lap_data",
]

QUALIFYING_COLUMNS = [
    "season",
    "round",
    "driver_id",
    "constructor_id",
    "quali_position",
    "q1_s",
    "q2_s",
    "q3_s",
    "representative_s",
]

SPRINT_COLUMNS = [
    "season",
    "round",
    "driver_id",
    "constructor_id",
    "finish_position",
    "grid",
    "status",
    "status_category",
]


def _frame(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[columns]


def attach_availability(
    results: pd.DataFrame,
    qualifying: pd.DataFrame,
    sprints: pd.DataFrame,
) -> pd.DataFrame:
    """Add data-availability flags. Missing qualifying stays missing."""
    frame = results.copy()
    if frame.empty:
        return _frame([], RESULT_COLUMNS)

    frame["status_category"] = frame["status"].map(classify_status)
    grid = pd.to_numeric(frame["grid"], errors="coerce")
    frame["has_grid"] = grid.ge(1).fillna(False)
    frame["has_race_time"] = pd.to_numeric(frame["race_time_ms"], errors="coerce").notna()
    frame["has_lap_data"] = False

    if qualifying.empty:
        quali_time = pd.Series(False, index=frame.index)
    else:
        quali = qualifying.drop_duplicates(["season", "round", "driver_id"], keep="first").copy()
        quali["has_time"] = pd.to_numeric(quali["representative_s"], errors="coerce").notna()
        keyed = quali.groupby(["season", "round", "driver_id"], as_index=False)["has_time"].any()
        frame = frame.merge(keyed, on=["season", "round", "driver_id"], how="left")
        quali_time = frame["has_time"].fillna(False).astype(bool)
        frame = frame.drop(columns=["has_time"])
    frame["has_qualifying_time"] = quali_time

    if sprints.empty:
        sprint_rounds: set[tuple[int, int]] = set()
    else:
        sprint_rounds = set(zip(sprints["season"].astype(int), sprints["round"].astype(int)))
    frame["has_sprint"] = [
        (int(season), int(rnd)) in sprint_rounds
        for season, rnd in zip(frame["season"], frame["round"])
    ]
    return frame


def _dedupe_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return results
    return results.drop_duplicates(["season", "round", "driver_id"], keep="first")


def save_processed(paths: DataPaths, tables: dict[str, pd.DataFrame]) -> None:
    paths.processed.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_parquet(paths.processed / f"{name}.parquet", index=False)


def load_processed(paths: DataPaths) -> dict[str, pd.DataFrame]:
    tables = {}
    for name in ("races", "results", "qualifying", "status", "sprints", "drivers", "constructors", "circuits"):
        file = paths.processed / f"{name}.parquet"
        if not file.exists():
            raise FileNotFoundError(
                f"Missing {file}. Run `f1champ ingest` before building features."
            )
        tables[name] = pd.read_parquet(file)
    return tables


def build_tables_from_pages(
    race_pages: list[dict],
    result_pages: list[dict],
    qualifying_pages: list[dict],
    sprint_pages: list[dict],
    status_pages: list[dict],
    driver_pages: list[dict],
    constructor_pages: list[dict],
    circuit_pages: list[dict],
) -> dict[str, pd.DataFrame]:
    races = pd.DataFrame(parse_races(race_pages))
    results = pd.DataFrame(parse_results(result_pages))
    qualifying = _frame(parse_qualifying(qualifying_pages), QUALIFYING_COLUMNS)
    qualifying = qualifying.drop_duplicates(["season", "round", "driver_id"], keep="first")
    sprints = _frame(parse_sprints(sprint_pages), SPRINT_COLUMNS)
    if not sprints.empty:
        sprints["status_category"] = sprints["status"].map(classify_status)
        sprints = sprints.drop_duplicates(["season", "round", "driver_id"], keep="first")
    results = _dedupe_results(results)
    results = attach_availability(results, qualifying, sprints)
    results = _frame(results.to_dict(orient="records"), RESULT_COLUMNS) if not results.empty else _frame([], RESULT_COLUMNS)

    status = pd.DataFrame(parse_status_table(status_pages))
    if status.empty:
        status = pd.DataFrame(columns=["status_id", "status", "count", "status_category"])
    else:
        status["status_category"] = status["status"].map(classify_status)
    drivers = pd.DataFrame(parse_drivers(driver_pages))
    if drivers.empty:
        drivers = pd.DataFrame(
            columns=["driver_id", "code", "permanent_number", "given_name", "family_name", "date_of_birth", "nationality"]
        )
    constructors = pd.DataFrame(parse_constructors(constructor_pages))
    if constructors.empty:
        constructors = pd.DataFrame(columns=["constructor_id", "name", "nationality"])
    circuits = pd.DataFrame(parse_circuits(circuit_pages))
    if circuits.empty:
        circuits = pd.DataFrame(columns=["circuit_id", "circuit_name", "locality", "country"])
    if races.empty:
        races = pd.DataFrame(
            columns=["season", "round", "race_name", "circuit_id", "circuit_name", "race_date", "race_time"]
        )
    return {
        "races": races,
        "results": results,
        "qualifying": qualifying,
        "status": status,
        "sprints": sprints,
        "drivers": drivers,
        "constructors": constructors,
        "circuits": circuits,
    }


def ingest(
    paths: DataPaths,
    start_season: int = START_SEASON,
    end_season: int | None = None,
    sleep_s: float = 0.2,
    client: JolpicaClient | None = None,
) -> dict[str, pd.DataFrame]:
    """Download Jolpica pages into data/raw and write processed parquet tables."""
    end_season = end_season or date.today().year
    own_client = client is None
    if client is None:
        client = JolpicaClient(paths.raw, sleep_s=sleep_s)
    try:
        season_pages = client.get_pages("seasons")
        years = [year for year in parse_seasons(season_pages) if start_season <= year <= end_season]
        if not years:
            raise JolpicaError(f"No seasons returned between {start_season} and {end_season}")

        race_pages: list[dict] = []
        result_pages: list[dict] = []
        qualifying_pages: list[dict] = []
        sprint_pages: list[dict] = []
        for year in years:
            logger.info("Ingesting %s", year)
            race_pages.extend(client.get_pages(f"{year}/races"))
            result_pages.extend(client.get_pages(f"{year}/results"))
            try:
                qualifying_pages.extend(client.get_pages(f"{year}/qualifying"))
            except JolpicaError:
                logger.info("No qualifying payload for %s", year)
            if year >= 2021:
                try:
                    sprint_pages.extend(client.get_pages(f"{year}/sprint"))
                except JolpicaError:
                    logger.info("No sprint payload for %s", year)

        tables = build_tables_from_pages(
            race_pages,
            result_pages,
            qualifying_pages,
            sprint_pages,
            client.get_pages("status"),
            client.get_pages("drivers"),
            client.get_pages("constructors"),
            client.get_pages("circuits"),
        )
    finally:
        if own_client:
            client.close()
    save_processed(paths, tables)
    logger.info(
        "Wrote %s races and %s results",
        len(tables["races"]),
        len(tables["results"]),
    )
    return tables
