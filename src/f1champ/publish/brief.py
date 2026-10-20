"""Plain-text race briefing written beside the prediction JSON.

The briefing is for a person reading the forecast. It uses the post-qualifying
information set: the grid, history from before this round, and (from 2023) the
sprint, which is finished before Grand Prix qualifying. It never includes this
round's race result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from f1champ.constants import ELO_INIT, MODEL_VERSION, N_SIM

# 2021–2022 set the race grid on Friday and ran the sprint afterwards.
# From 2023 the sprint is over before Grand Prix qualifying.
SPRINT_BEFORE_RACE_QUALIFYING = 2023
_SPRINT_POINTS_EARLY = {1: 3, 2: 2, 3: 1}
_SPRINT_POINTS = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}


@dataclass
class _Names:
    drivers: dict[str, str] = field(default_factory=dict)
    codes: dict[str, str] = field(default_factory=dict)
    numbers: dict[str, str] = field(default_factory=dict)
    nationalities: dict[str, str] = field(default_factory=dict)
    constructors: dict[str, str] = field(default_factory=dict)

    def driver(self, driver_id: object) -> str:
        key = "" if driver_id is None or _missing(driver_id) else str(driver_id)
        if key in self.drivers:
            return self.drivers[key]
        return key.replace("_", " ").title() or "Unknown driver"

    def constructor(self, constructor_id: object) -> str:
        if constructor_id is None or _missing(constructor_id):
            return "Unknown team"
        key = str(constructor_id)
        if key in self.constructors:
            return self.constructors[key]
        return key.replace("_", " ").title()


def write_race_brief(
    directory: Path,
    predicted: pd.DataFrame,
    *,
    races: pd.DataFrame | None = None,
    results: pd.DataFrame | None = None,
    drivers: pd.DataFrame | None = None,
    constructors: pd.DataFrame | None = None,
    circuits: pd.DataFrame | None = None,
    sprints: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    n_sim: int = N_SIM,
) -> Path:
    """Write `{season}_{round}.txt` next to the JSON prediction."""
    text = render_race_brief(
        predicted,
        races=races,
        results=results,
        drivers=drivers,
        constructors=constructors,
        circuits=circuits,
        sprints=sprints,
        features=features,
        n_sim=n_sim,
    )
    season = int(predicted["season"].iloc[0])
    rnd = int(predicted["round"].iloc[0])
    path = Path(directory) / f"{season}_{rnd:02d}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def render_race_brief(
    predicted: pd.DataFrame,
    *,
    races: pd.DataFrame | None = None,
    results: pd.DataFrame | None = None,
    drivers: pd.DataFrame | None = None,
    constructors: pd.DataFrame | None = None,
    circuits: pd.DataFrame | None = None,
    sprints: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    n_sim: int = N_SIM,
) -> str:
    if predicted is None or predicted.empty:
        raise ValueError("Cannot write a briefing for an empty prediction.")
    names = _build_names(drivers, constructors)
    season = int(predicted["season"].iloc[0])
    rnd = int(predicted["round"].iloc[0])
    circuit_id = _first_text(predicted, "circuit_id")
    race_meta = _race_meta(races, season, rnd)
    if circuit_id is None:
        circuit_id = race_meta.get("circuit_id")
    circuit = _circuit_meta(circuits, race_meta, circuit_id)
    prior = _before_race(_frame(results), season, rnd)
    known_sprints = _known_sprints(_frame(sprints), season, rnd)
    this_sprint = _this_sprint(_frame(sprints), season, rnd)
    features_prior = _before_race(_frame(features), season, rnd)
    lines: list[str] = []
    lines.extend(_header(race_meta, circuit, season, rnd, predicted, n_sim))
    lines.extend(_prediction_section(predicted, names))
    driver_rows, constructor_rows = _standings(prior, known_sprints, predicted, season)
    lines.extend(_sprint_section(predicted, this_sprint, names, season))
    lines.extend(_championship_section(driver_rows, constructor_rows, known_sprints, predicted, names, season))
    lines.extend(
        _history_section(
            prior,
            features_prior,
            _frame(races),
            names,
            circuit_id,
            circuit["circuit_name"],
        )
    )
    lines.extend(
        _driver_sections(
            predicted,
            names,
            prior,
            features_prior,
            this_sprint,
            driver_rows,
            circuit_id,
            circuit["circuit_name"],
            season,
        )
    )
    lines.extend(_notes())
    return "\n".join(lines).rstrip() + "\n"


def _header(race_meta, circuit, season: int, rnd: int, predicted: pd.DataFrame, n_sim: int) -> list[str]:
    title = race_meta.get("race_name") or f"{season} round {rnd}"
    place = _place_line(circuit)
    when = _when_line(race_meta.get("race_date") or _first_text(predicted, "race_date"), race_meta.get("race_time"))
    lines = [str(title), f"{season} round {rnd}"]
    if place:
        lines.append(place)
    if when:
        lines.append(when)
    lines.extend(
        [
            "",
            (
                f"Post-qualifying forecast ({MODEL_VERSION}, {n_sim} simulations). "
                "Win, podium, and points chances count a retirement as a miss. "
                "The expected place is the average finish in the simulations where that driver is classified. "
                f"Driver and car ratings start at {ELO_INIT:.0f}."
            ),
            "This file does not include the result of this race.",
        ]
    )
    return lines


def _prediction_section(predicted: pd.DataFrame, names: _Names) -> list[str]:
    ordered = _by_win(predicted)
    lines = _section("Predicted race")
    if ordered.empty:
        lines.append("No drivers in the forecast.")
        return lines
    top = ordered.iloc[0]
    lines.append(
        f"Most likely winner: {names.driver(top['driver_id'])} "
        f"({names.constructor(top['constructor_id'])}), {_pct(top['win_probability'])}."
    )
    lines.append("")
    lines.append("Highest podium chances. Each figure is that driver's own chance, not the chance of this exact trio.")
    for index, (_, row) in enumerate(ordered.head(3).iterrows(), start=1):
        lines.append(
            f"  {index}. {names.driver(row['driver_id'])} ({names.constructor(row['constructor_id'])}), "
            f"{_pct(row['podium_probability'])} podium, {_pct(row['win_probability'])} win, "
            f"expected {_place_number(row['expected_position'])} if classified."
        )
    lines.append("")
    headers = ["#", "Driver", "Team", "Grid", "Qual", "Win", "Podium", "Points", "Retire", "Exp."]
    rows = []
    for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
        rows.append(
            [
                str(rank),
                names.driver(row["driver_id"]),
                names.constructor(row["constructor_id"]),
                _grid(row.get("grid_position")),
                _grid(row.get("quali_position")),
                _pct(row.get("win_probability")),
                _pct(row.get("podium_probability")),
                _pct(_points_probability(row)),
                _pct(row.get("retirement_probability")),
                _place_number(row.get("expected_position")),
            ]
        )
    lines.extend(_table(headers, rows, aligns=["r", "l", "l", "r", "r", "r", "r", "r", "r", "r"]))
    lines.append("")
    lines.append("Exp. is the expected classified finish. Points is a top-10 finish after retirements.")
    return lines


def _sprint_section(predicted: pd.DataFrame, this_sprint: pd.DataFrame, names: _Names, season: int) -> list[str]:
    sprint_weekend = _flag(predicted, "sprint_held") or _flag(predicted, "has_sprint")
    if this_sprint.empty and not sprint_weekend:
        return []
    lines = _section("Sprint this weekend")
    if season < SPRINT_BEFORE_RACE_QUALIFYING:
        lines.append(
            "This season ran the sprint after Grand Prix qualifying, so the sprint result is not known to this forecast."
        )
        return lines
    if this_sprint.empty:
        lines.append("This is a sprint weekend. No sprint result is stored yet.")
        return lines
    lines.append("The sprint was run before Grand Prix qualifying, so it is known for this forecast.")
    ordered = this_sprint.copy()
    ordered["_place"] = pd.to_numeric(ordered["finish_position"], errors="coerce")
    ordered = ordered.sort_values(["_place", "driver_id"], na_position="last")
    rows = []
    for _, row in ordered.iterrows():
        rows.append(
            [
                _finish_label(row),
                names.driver(row["driver_id"]),
                names.constructor(row.get("constructor_id")),
            ]
        )
    lines.extend(_table(["Result", "Driver", "Team"], rows, aligns=["l", "l", "l"]))
    return lines


def _championship_section(
    driver_rows: list[dict],
    constructor_rows: list[dict],
    known_sprints: pd.DataFrame,
    predicted: pd.DataFrame,
    names: _Names,
    season: int,
) -> list[str]:
    lines = _section("Championship entering the race")
    if not driver_rows:
        lines.append("No championship points before this race.")
        return lines
    lines.append("Race points from earlier rounds this season, plus sprint points already on the board.")
    if _current_sprint_counts(known_sprints, season, int(predicted["round"].iloc[0])):
        lines.append("That includes this weekend's sprint.")
    lines.append("")
    lines.append("Drivers")
    lines.extend(
        _table(
            ["Pos", "Driver", "Pts", "Wins", "Podiums"],
            [
                [str(index), names.driver(row["driver_id"]), _points(row["points"]), str(row["wins"]), str(row["podiums"])]
                for index, row in enumerate(driver_rows, start=1)
            ],
            aligns=["r", "l", "r", "r", "r"],
        )
    )
    if constructor_rows:
        lines.append("")
        lines.append("Teams")
        lines.extend(
            _table(
                ["Pos", "Team", "Pts"],
                [
                    [str(index), names.constructor(row["constructor_id"]), _points(row["points"])]
                    for index, row in enumerate(constructor_rows, start=1)
                ],
                aligns=["r", "l", "r"],
            )
        )
    lines.append("Tied drivers are ordered by wins, then podiums.")
    return lines


def _history_section(
    prior: pd.DataFrame,
    features_prior: pd.DataFrame,
    races: pd.DataFrame,
    names: _Names,
    circuit_id: str | None,
    circuit_name: str,
) -> list[str]:
    title = f"Previous winners at {circuit_name}" if circuit_name else "Previous winners at this circuit"
    lines = _section(title)
    if not circuit_id or prior.empty or "circuit_id" not in prior.columns:
        lines.append("No earlier grands prix at this circuit are in the data.")
        return lines
    visits = prior[prior["circuit_id"].astype(str) == str(circuit_id)].copy()
    if visits.empty:
        lines.append("No earlier grands prix at this circuit are in the data.")
        return lines
    places = pd.to_numeric(visits["finish_position"], errors="coerce")
    winners = visits[places.eq(1)].copy()
    if winners.empty:
        lines.append("Earlier visits are stored, but none has a winner recorded.")
        return lines
    winners = winners.sort_values(["season", "round"])
    race_names = _race_name_map(races)
    pole_names = _pole_map(features_prior, visits, names)
    counts: dict[str, int] = {}
    rows = []
    for _, row in winners.iterrows():
        driver = names.driver(row["driver_id"])
        counts[driver] = counts.get(driver, 0) + 1
        key = (int(row["season"]), int(row["round"]))
        rows.append(
            [
                str(int(row["season"])),
                race_names.get(key, ""),
                driver,
                names.constructor(row.get("constructor_id")),
                pole_names.get(key, "unknown"),
            ]
        )
    years = [int(value) for value in winners["season"]]
    lines.append(f"{len(rows)} grands prix in this data, {min(years)}–{max(years)}.")
    lines.append(_win_summary(counts))
    lines.append("")
    lines.extend(_table(["Year", "Race", "Winner", "Team", "Pole"], rows, aligns=["r", "l", "l", "l", "l"]))
    return lines


def _driver_sections(
    predicted: pd.DataFrame,
    names: _Names,
    prior: pd.DataFrame,
    features_prior: pd.DataFrame,
    this_sprint: pd.DataFrame,
    driver_rows: list[dict],
    circuit_id: str | None,
    circuit_name: str,
    season: int,
) -> list[str]:
    lines = _section("Drivers")
    ordered = _by_win(predicted)
    teammates = _teammates(ordered)
    circuit_label = circuit_name or "this circuit"
    for rank, (_, row) in enumerate(ordered.iterrows(), start=1):
        if rank > 1:
            lines.append("")
        lines.extend(
            _one_driver(
                rank,
                row,
                names,
                prior,
                features_prior,
                this_sprint,
                teammates,
                driver_rows,
                circuit_id,
                circuit_label,
                season,
            )
        )
    return lines


def _one_driver(
    rank: int,
    row: pd.Series,
    names: _Names,
    prior: pd.DataFrame,
    features_prior: pd.DataFrame,
    this_sprint: pd.DataFrame,
    teammates: dict[str, list[str]],
    driver_rows: list[dict],
    circuit_id: str | None,
    circuit_label: str,
    season: int,
) -> list[str]:
    driver_id = str(row["driver_id"])
    name = names.driver(driver_id)
    identity = _identity(driver_id, names)
    lines = [f"{rank}. {name}" + (f"  ({identity})" if identity else "")]
    lines.append(names.constructor(row.get("constructor_id")))
    lines.append(_grid_sentence(row))
    sprint_line = _sprint_sentence(this_sprint, driver_id)
    if sprint_line:
        lines.append(sprint_line)
    place, chance = _most_likely(row.get("position_distribution"))
    lines.append(
        "Forecast: "
        f"{_pct(row.get('win_probability'))} win, "
        f"{_pct(row.get('podium_probability'))} podium, "
        f"{_pct(_points_probability(row))} points, "
        f"{_pct(row.get('retirement_probability'))} retirement "
        f"({_pct(row.get('mechanical_retirement_probability'))} mechanical)."
    )
    classified = f"If classified, expected {_place_number(row.get('expected_position'))}."
    if place:
        classified += f" Most likely place if classified: {place} ({_pct(chance)} of those finishes)."
    lines.append(classified)
    lines.extend(_season_lines(prior, driver_id, names, teammates.get(driver_id, []), driver_rows, season, row))
    lines.extend(_circuit_lines(prior, features_prior, driver_id, circuit_id, circuit_label))
    lines.extend(_rating_lines(row))
    return lines


def _season_lines(
    prior: pd.DataFrame,
    driver_id: str,
    names: _Names,
    teammate_ids: list[str],
    driver_rows: list[dict],
    season: int,
    row: pd.Series,
) -> list[str]:
    lines = []
    personal = _driver_results(prior, driver_id)
    season_rows = personal[personal["season"].astype(int) == season] if not personal.empty else personal
    standing = _standing_phrase(driver_rows, driver_id)
    if season_rows.empty:
        if standing:
            lines.append(f"This season, before this race: {standing}. No grands prix before this round.")
        else:
            lines.append("No starts recorded before this race in the current season.")
    else:
        lines.append(f"This season, before this race: {standing}. Grands prix: {_record_summary(season_rows)}.")
    recent = _recent_phrase(personal, season)
    if recent:
        lines.append(f"Last five grands prix: {recent}.")
    for teammate_id in teammate_ids:
        phrase = _head_to_head(prior, driver_id, teammate_id, season)
        if phrase:
            lines.append(f"Against {names.driver(teammate_id)} this season: {phrase}.")
    gap_days = row.get("days_since_last_race")
    if gap_days is not None and not _missing(gap_days):
        days = int(round(float(gap_days)))
        lines.append(f"{days} day{'s' if days != 1 else ''} since the previous grand prix.")
    return lines


def _circuit_lines(
    prior: pd.DataFrame,
    features_prior: pd.DataFrame,
    driver_id: str,
    circuit_id: str | None,
    circuit_label: str,
) -> list[str]:
    if not circuit_id or prior.empty or "circuit_id" not in prior.columns:
        return [f"No earlier starts at {circuit_label}."]
    visits = _driver_results(prior, driver_id)
    visits = visits[visits["circuit_id"].astype(str) == str(circuit_id)]
    if visits.empty:
        return [f"No earlier starts at {circuit_label}."]
    summary = _record_summary(visits)
    poles = _driver_poles(features_prior, driver_id, circuit_id)
    pole_text = f", {poles} pole{'s' if poles != 1 else ''}" if poles else ""
    lines = [f"At {circuit_label}: {summary}{pole_text}."]
    last = visits.sort_values(["season", "round"]).iloc[-1]
    lines.append(f"Last visit: {int(last['season'])}, {_finish_label(last)}.")
    return lines


def _rating_lines(row: pd.Series) -> list[str]:
    lines = []
    driver_elo = _optional_float(row.get("driver_elo"))
    car_elo = _optional_float(row.get("constructor_elo"))
    if driver_elo is not None or car_elo is not None:
        parts = []
        if driver_elo is not None:
            parts.append(f"driver {_elo_phrase(driver_elo)}")
        if car_elo is not None:
            parts.append(f"car {_elo_phrase(car_elo)}")
        lines.append("Ratings coming in: " + ", ".join(parts) + ".")
    teammate = _optional_float(row.get("teammate_residual"))
    if teammate is not None:
        lines.append(_teammate_phrase(teammate))
    circuit = _optional_float(row.get("circuit_residual"))
    if circuit is not None:
        lines.append(_circuit_phrase(circuit))
    return lines


def _notes() -> list[str]:
    lines = _section("Notes")
    lines.append("The published grid can include corrections applied after the start.")
    lines.append("Retirements are estimated separately for each driver, so one driver's retirement does not change another's chance.")
    lines.append("Lap pace, weather, and pit strategy are not in this forecast.")
    lines.append("Expected place ignores the simulations in which that driver does not finish.")
    lines.append("History stops before this round. From 2023 the sprint is listed because it is already finished before Grand Prix qualifying.")
    return lines


def _standings(
    prior: pd.DataFrame,
    known_sprints: pd.DataFrame,
    predicted: pd.DataFrame,
    season: int,
) -> tuple[list[dict], list[dict]]:
    drivers: dict[str, dict] = {}
    constructors: dict[str, dict] = {}

    def driver_bucket(driver_id: str) -> dict:
        return drivers.setdefault(driver_id, {"driver_id": driver_id, "points": 0.0, "wins": 0, "podiums": 0, "starts": 0})

    def team_bucket(constructor_id: str) -> dict:
        return constructors.setdefault(constructor_id, {"constructor_id": constructor_id, "points": 0.0})

    season_results = _season_slice(prior, season)
    if not season_results.empty:
        for row in season_results.to_dict(orient="records"):
            driver_id = str(row["driver_id"])
            bucket = driver_bucket(driver_id)
            bucket["starts"] += 1
            bucket["points"] += _optional_float(row.get("points")) or 0.0
            if _is_win(row):
                bucket["wins"] += 1
            if _is_podium(row):
                bucket["podiums"] += 1
            constructor_id = row.get("constructor_id")
            if constructor_id is not None and not _missing(constructor_id):
                team_bucket(str(constructor_id))["points"] += _optional_float(row.get("points")) or 0.0

    season_sprints = _season_slice(known_sprints, season)
    if not season_sprints.empty:
        for row in season_sprints.to_dict(orient="records"):
            awarded = sprint_points(season, row.get("finish_position"))
            if awarded:
                driver_bucket(str(row["driver_id"]))["points"] += awarded
                constructor_id = row.get("constructor_id")
                if constructor_id is not None and not _missing(constructor_id):
                    team_bucket(str(constructor_id))["points"] += awarded

    for driver_id in predicted["driver_id"].astype(str):
        driver_bucket(driver_id)
    if all(row["starts"] == 0 and row["points"] == 0 for row in drivers.values()):
        return [], []
    driver_rows = sorted(
        drivers.values(),
        key=lambda row: (-row["points"], -row["wins"], -row["podiums"], row["driver_id"]),
    )
    constructor_rows = sorted(constructors.values(), key=lambda row: (-row["points"], row["constructor_id"]))
    return driver_rows, constructor_rows


def _standing_phrase(driver_rows: list[dict], driver_id: str) -> str:
    for index, row in enumerate(driver_rows, start=1):
        if row["driver_id"] == driver_id:
            return f"{_ordinal(index)} with {_points(row['points'])} points"
    return ""


def _current_sprint_counts(known_sprints: pd.DataFrame, season: int, rnd: int) -> bool:
    if known_sprints.empty or season < SPRINT_BEFORE_RACE_QUALIFYING:
        return False
    seasons = known_sprints["season"].astype(int)
    rounds = known_sprints["round"].astype(int)
    return bool(((seasons == season) & (rounds == rnd)).any())


def sprint_points(season: int, position) -> float:
    if position is None or _missing(position):
        return 0.0
    place = int(position)
    if season < 2021:
        return 0.0
    table = _SPRINT_POINTS_EARLY if season <= 2022 else _SPRINT_POINTS
    return float(table.get(place, 0))


def _known_sprints(sprints: pd.DataFrame, season: int, rnd: int) -> pd.DataFrame:
    """Sprints whose result is knowable at the post-qualifying cutoff."""
    if sprints.empty:
        return sprints
    known = _before_race(sprints, season, rnd)
    if season >= SPRINT_BEFORE_RACE_QUALIFYING:
        current = _this_sprint(sprints, season, rnd)
        if not current.empty:
            known = pd.concat([known, current], ignore_index=True)
    return known


def _this_sprint(sprints: pd.DataFrame, season: int, rnd: int) -> pd.DataFrame:
    if sprints.empty or season < SPRINT_BEFORE_RACE_QUALIFYING:
        return sprints.iloc[0:0]
    mask = (sprints["season"].astype(int) == season) & (sprints["round"].astype(int) == rnd)
    return sprints.loc[mask].copy()


def _before_race(frame: pd.DataFrame, season: int, rnd: int) -> pd.DataFrame:
    if frame.empty or "season" not in frame.columns or "round" not in frame.columns:
        return frame.iloc[0:0]
    seasons = frame["season"].astype(int)
    rounds = frame["round"].astype(int)
    return frame.loc[(seasons < season) | ((seasons == season) & (rounds < rnd))].copy()


def _season_slice(frame: pd.DataFrame, season: int) -> pd.DataFrame:
    if frame.empty or "season" not in frame.columns:
        return frame.iloc[0:0]
    return frame.loc[frame["season"].astype(int) == season].copy()


def _driver_results(frame: pd.DataFrame, driver_id: str) -> pd.DataFrame:
    if frame.empty or "driver_id" not in frame.columns:
        return frame.iloc[0:0]
    return frame.loc[frame["driver_id"].astype(str) == str(driver_id)].copy()


def _record_summary(frame: pd.DataFrame) -> str:
    starts = len(frame)
    wins = int(frame.apply(_is_win, axis=1).sum()) if starts else 0
    podiums = int(frame.apply(_is_podium, axis=1).sum()) if starts else 0
    classified = frame[frame.apply(_is_classified, axis=1)] if starts else frame
    positions = pd.to_numeric(classified["finish_position"], errors="coerce") if starts else pd.Series(dtype=float)
    best = None
    average = None
    if not positions.empty and positions.notna().any():
        best = int(positions.min())
        average = float(positions.mean())
    retirements = int(frame.apply(_retired, axis=1).sum()) if starts else 0
    parts = [f"{starts} start{'s' if starts != 1 else ''}", f"{wins} win{'s' if wins != 1 else ''}", f"{podiums} podium{'s' if podiums != 1 else ''}"]
    if best is not None:
        parts.append(f"best {_ordinal(best)}")
    if average is not None:
        parts.append(f"average classified {_fmt_float(average)}")
    if retirements:
        parts.append(f"{retirements} retirement{'s' if retirements != 1 else ''}")
    return ", ".join(parts)


def _recent_phrase(personal: pd.DataFrame, season: int) -> str:
    if personal.empty:
        return ""
    recent = personal.sort_values(["season", "round"]).tail(5)
    parts = []
    for _, row in recent.iloc[::-1].iterrows():
        when = f"R{int(row['round'])}"
        if int(row["season"]) != season:
            when = f"{int(row['season'])} {when}"
        parts.append(f"{when} {_finish_label(row)}")
    return ", ".join(parts)


def _head_to_head(prior: pd.DataFrame, driver_id: str, teammate_id: str, season: int) -> str:
    if prior.empty:
        return ""
    season_rows = _season_slice(prior, season)
    if season_rows.empty or "constructor_id" not in season_rows.columns:
        return ""
    mine = _driver_results(season_rows, driver_id)
    theirs = _driver_results(season_rows, teammate_id)
    if mine.empty or theirs.empty:
        return ""
    merged = mine.merge(theirs, on=["season", "round"], suffixes=("", "_mate"))
    if merged.empty:
        return ""
    same_team = merged["constructor_id"].astype(str) == merged["constructor_id_mate"].astype(str)
    merged = merged.loc[same_team]
    if merged.empty:
        return ""
    ahead = behind = unfinished = 0
    for _, row in merged.iterrows():
        both = _is_classified(row) and _is_classified(row, suffix="_mate")
        if not both:
            unfinished += 1
            continue
        mine_pos = pd.to_numeric(row.get("finish_position"), errors="coerce")
        mate_pos = pd.to_numeric(row.get("finish_position_mate"), errors="coerce")
        if pd.isna(mine_pos) or pd.isna(mate_pos) or mine_pos == mate_pos:
            continue
        if mine_pos < mate_pos:
            ahead += 1
        else:
            behind += 1
    parts = [f"ahead {ahead}", f"behind {behind}"]
    if unfinished:
        noun = "race" if unfinished == 1 else "races"
        parts.append(f"{unfinished} {noun} where one of them did not finish")
    return ", ".join(parts)


def _driver_poles(features_prior: pd.DataFrame, driver_id: str, circuit_id: str) -> int:
    if features_prior.empty or "quali_position" not in features_prior.columns or "circuit_id" not in features_prior.columns:
        return 0
    rows = _driver_results(features_prior, driver_id)
    rows = rows[rows["circuit_id"].astype(str) == str(circuit_id)]
    quali = pd.to_numeric(rows["quali_position"], errors="coerce")
    return int(quali.eq(1).sum())


def _pole_map(features_prior: pd.DataFrame, visits: pd.DataFrame, names: _Names) -> dict[tuple[int, int], str]:
    poles: dict[tuple[int, int], str] = {}
    keys = {(int(season), int(rnd)) for season, rnd in zip(visits["season"], visits["round"])}
    if not features_prior.empty and "quali_position" in features_prior.columns:
        quali = pd.to_numeric(features_prior["quali_position"], errors="coerce")
        pole_rows = features_prior.loc[quali.eq(1)]
        for _, row in pole_rows.iterrows():
            key = (int(row["season"]), int(row["round"]))
            if key in keys:
                poles[key] = names.driver(row["driver_id"])
    if "grid" in visits.columns:
        grid = pd.to_numeric(visits["grid"], errors="coerce")
        for _, row in visits.loc[grid.eq(1)].iterrows():
            key = (int(row["season"]), int(row["round"]))
            poles.setdefault(key, names.driver(row["driver_id"]))
    return poles


def _race_name_map(races: pd.DataFrame) -> dict[tuple[int, int], str]:
    if races.empty or "race_name" not in races.columns:
        return {}
    mapping = {}
    for _, row in races.iterrows():
        if _missing(row.get("season")) or _missing(row.get("round")):
            continue
        mapping[(int(row["season"]), int(row["round"]))] = _text(row.get("race_name")) or ""
    return mapping


def _win_summary(counts: dict[str, int]) -> str:
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    multiple = [f"{name} {count}" for name, count in ordered if count > 1]
    single = [name for name, count in ordered if count == 1]
    parts = []
    if multiple:
        parts.append("More than one win: " + ", ".join(multiple) + ".")
    if single:
        parts.append("One win: " + ", ".join(single) + ".")
    return " ".join(parts)


def _build_names(drivers: pd.DataFrame | None, constructors: pd.DataFrame | None) -> _Names:
    names = _Names()
    driver_frame = _frame(drivers)
    if not driver_frame.empty and "driver_id" in driver_frame.columns:
        for row in driver_frame.drop_duplicates("driver_id").to_dict(orient="records"):
            key = str(row["driver_id"])
            given = _text(row.get("given_name"))
            family = _text(row.get("family_name"))
            full = " ".join(part for part in (given, family) if part)
            names.drivers[key] = full or key.replace("_", " ").title()
            code = _text(row.get("code"))
            if code:
                names.codes[key] = code
            number = _number_text(row.get("permanent_number"))
            if number:
                names.numbers[key] = number
            nationality = _text(row.get("nationality"))
            if nationality:
                names.nationalities[key] = nationality
    constructor_frame = _frame(constructors)
    if not constructor_frame.empty and "constructor_id" in constructor_frame.columns:
        for row in constructor_frame.drop_duplicates("constructor_id").to_dict(orient="records"):
            label = _text(row.get("name"))
            names.constructors[str(row["constructor_id"])] = label or str(row["constructor_id"])
    return names


def _race_meta(races: pd.DataFrame | None, season: int, rnd: int) -> dict:
    frame = _frame(races)
    if frame.empty:
        return {}
    match = frame[(frame["season"].astype(int) == season) & (frame["round"].astype(int) == rnd)]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _circuit_meta(circuits: pd.DataFrame | None, race_meta: dict, circuit_id: str | None) -> dict:
    meta = {
        "circuit_name": _text(race_meta.get("circuit_name")) or "",
        "locality": "",
        "country": "",
    }
    frame = _frame(circuits)
    if circuit_id and not frame.empty and "circuit_id" in frame.columns:
        match = frame[frame["circuit_id"].astype(str) == str(circuit_id)]
        if not match.empty:
            row = match.iloc[0]
            meta["circuit_name"] = _text(row.get("circuit_name")) or meta["circuit_name"]
            meta["locality"] = _text(row.get("locality")) or ""
            meta["country"] = _text(row.get("country")) or ""
    return meta


def _place_line(circuit: dict) -> str:
    name = circuit.get("circuit_name") or ""
    locality = circuit.get("locality") or ""
    country = circuit.get("country") or ""
    extras = []
    if locality and locality.lower() not in name.lower():
        extras.append(locality)
    if country and country.lower() not in name.lower():
        extras.append(country)
    if name and extras:
        return f"{name}, {', '.join(extras)}"
    return name or ", ".join(extras)


def _when_line(race_date, race_time) -> str:
    stamp = pd.to_datetime(race_date, errors="coerce")
    if pd.isna(stamp):
        date_text = ""
    else:
        date_text = f"{stamp.strftime('%A')} {int(stamp.day)} {stamp.strftime('%B %Y')}"
    clock = _clock(race_time)
    if date_text and clock:
        return f"{date_text}, {clock}"
    return date_text or clock


def _clock(race_time) -> str:
    text = _text(race_time)
    if not text:
        return ""
    cleaned = text.replace("Z", "").strip()
    parts = cleaned.split(":")
    if len(parts) >= 2 and parts[0].isdigit():
        return f"{int(parts[0]):02d}:{int(parts[1]):02d} UTC"
    return text


def _grid_sentence(row: pd.Series) -> str:
    grid = _optional_int(row.get("grid_position"))
    quali = _optional_int(row.get("quali_position"))
    gap = _optional_float(row.get("quali_gap_s"))
    gap_text = _gap_text(gap)
    if grid and quali and grid != quali:
        sentence = f"Starts {_ordinal(grid)} after qualifying {_ordinal(quali)}"
    elif grid:
        sentence = f"Starts {_ordinal(grid)}"
    elif quali:
        sentence = f"Qualified {_ordinal(quali)}"
    else:
        sentence = "Grid unknown"
    if gap_text:
        sentence += f", {gap_text}"
    return sentence + "."


def _gap_text(gap: float | None) -> str:
    if gap is None:
        return ""
    if abs(gap) < 0.0005:
        return "pole time"
    return f"{gap:+.3f}s to pole"


def _sprint_sentence(this_sprint: pd.DataFrame, driver_id: str) -> str:
    if this_sprint.empty:
        return ""
    rows = _driver_results(this_sprint, driver_id)
    if rows.empty:
        return "No sprint result stored for this driver."
    row = rows.iloc[0]
    return f"Sprint: {_finish_label(row)}."


def _identity(driver_id: str, names: _Names) -> str:
    parts = []
    code = names.codes.get(driver_id)
    number = names.numbers.get(driver_id)
    nationality = names.nationalities.get(driver_id)
    if code:
        parts.append(code)
    if number:
        parts.append(f"#{number}")
    if nationality:
        parts.append(nationality)
    return ", ".join(parts)


def _teammate_phrase(residual: float) -> str:
    if abs(residual) < 0.3:
        return "Recent results against teammates are about level."
    direction = "ahead of" if residual > 0 else "behind"
    return f"Recent results put them about {abs(residual):.1f} places {direction} their teammates."


def _circuit_phrase(residual: float) -> str:
    if abs(residual) < 0.25:
        return "This car's results here are in line with its results at other circuits."
    if residual > 0:
        return f"This car has finished about {residual:.1f} places further back here than at other circuits."
    return f"This car has finished about {abs(residual):.1f} places further forward here than at other circuits."


def _elo_phrase(value: float) -> str:
    delta = value - ELO_INIT
    if abs(delta) < 15:
        relation = "about average"
    elif delta > 0:
        relation = f"{delta:.0f} above average"
    else:
        relation = f"{abs(delta):.0f} below average"
    return f"{value:.0f} ({relation})"


def _teammates(predicted: pd.DataFrame) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    if "constructor_id" not in predicted.columns:
        return grouped
    teams: dict[str, list[str]] = {}
    for _, row in predicted.iterrows():
        if _missing(row.get("constructor_id")):
            continue
        teams.setdefault(str(row["constructor_id"]), []).append(str(row["driver_id"]))
    for members in teams.values():
        for driver_id in members:
            grouped[driver_id] = [other for other in members if other != driver_id]
    return grouped


def _by_win(predicted: pd.DataFrame) -> pd.DataFrame:
    frame = predicted.copy()
    frame["_win"] = pd.to_numeric(frame["win_probability"], errors="coerce").fillna(-1)
    frame["_exp"] = pd.to_numeric(frame["expected_position"], errors="coerce").fillna(999)
    frame["_id"] = frame["driver_id"].astype(str)
    return frame.sort_values(["_win", "_exp", "_id"], ascending=[False, True, True]).drop(columns=["_win", "_exp", "_id"])


def _points_probability(row: pd.Series) -> float | None:
    distribution = row.get("position_distribution")
    retirement = _optional_float(row.get("retirement_probability"))
    if distribution is None or retirement is None:
        return None
    values = np.asarray(list(distribution) if isinstance(distribution, tuple) else distribution, dtype=float)
    if values.size == 0:
        return None
    classified_points = float(values[: min(10, values.size)].sum())
    return (1.0 - retirement) * classified_points


def _most_likely(distribution) -> tuple[str, float] | tuple[None, None]:
    if distribution is None:
        return None, None
    values = np.asarray(list(distribution) if isinstance(distribution, tuple) else distribution, dtype=float)
    if values.size == 0 or not np.isfinite(values).any():
        return None, None
    index = int(np.nanargmax(values))
    return _ordinal(index + 1), float(values[index])


def _finish_label(row) -> str:
    position = row.get("finish_position")
    if _is_classified(row) and not _missing(position):
        return _ordinal(int(position))
    status = _text(row.get("status"))
    if status:
        return status
    category = _text(row.get("status_category"))
    return category or "no result"


def _is_classified(row, suffix: str = "") -> bool:
    category = row.get(f"status_category{suffix}")
    if category is not None and not _missing(category):
        return str(category) == "finished"
    status = _text(row.get(f"status{suffix}")).lower()
    return status in {"finished", "lapped"} or status.startswith("+")


def _retired(row) -> bool:
    category = _text(row.get("status_category")).lower()
    if category in {"mechanical", "incident", "unknown", "disqualified"}:
        return True
    if category == "finished" or _is_classified(row):
        return False
    status = _text(row.get("status")).lower()
    return bool(status) and status not in {"finished", "lapped"}


def _is_win(row) -> bool:
    position = pd.to_numeric(row.get("finish_position"), errors="coerce")
    return bool(pd.notna(position) and int(position) == 1 and _is_classified(row))


def _is_podium(row) -> bool:
    position = pd.to_numeric(row.get("finish_position"), errors="coerce")
    return bool(pd.notna(position) and int(position) <= 3 and _is_classified(row))


def _section(title: str) -> list[str]:
    return ["", title, "-" * len(title)]


def _table(headers: list[str], rows: list[list[str]], aligns: list[str]) -> list[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def format_row(row: list[str]) -> str:
        cells = []
        for index, cell in enumerate(row):
            cells.append(cell.rjust(widths[index]) if aligns[index] == "r" else cell.ljust(widths[index]))
        return "  ".join(cells).rstrip()

    lines = [format_row(headers), format_row(["-" * width for width in widths])]
    lines.extend(format_row(row) for row in rows)
    return lines


def _frame(value: pd.DataFrame | None) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    return value


def _flag(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame.columns:
        return False
    values = frame[column]
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().any():
        return bool(numeric.fillna(0).gt(0).any())
    return bool(values.fillna(False).astype(bool).any())


def _first_text(frame: pd.DataFrame, column: str) -> str | None:
    if column not in frame.columns or frame.empty:
        return None
    return _text(frame[column].iloc[0]) or None


def _pct(value) -> str:
    number = _optional_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.1f}%"


def _points(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"


def _place_number(value) -> str:
    number = _optional_float(value)
    if number is None:
        return "n/a"
    return _fmt_float(number)


def _fmt_float(value: float) -> str:
    return f"{value:.1f}"


def _ordinal(number: int) -> str:
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def _grid(value) -> str:
    number = _optional_int(value)
    return str(number) if number is not None else "-"


def _optional_int(value) -> int | None:
    number = _optional_float(value)
    if number is None:
        return None
    return int(round(number))


def _optional_float(value) -> float | None:
    if value is None or _missing(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number) or np.isinf(number):
        return None
    return number


def _number_text(value) -> str:
    if value is None or _missing(value):
        return ""
    number = _optional_float(value)
    if number is not None and float(number).is_integer():
        return str(int(number))
    return str(value).strip()


def _text(value) -> str:
    if value is None or _missing(value):
        return ""
    return str(value).strip()


def _missing(value) -> bool:
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, bool):
        return missing
    return False
