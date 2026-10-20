from pathlib import Path

import numpy as np
import pandas as pd

from f1champ.publish.brief import render_race_brief, sprint_points, write_race_brief


def _prediction(season: int, rnd: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season": season,
                "round": rnd,
                "race_date": "2024-09-01",
                "circuit_id": "monza",
                "driver_id": "a",
                "constructor_id": "red",
                "grid_position": 1,
                "quali_position": 1,
                "quali_gap_s": 0.0,
                "driver_elo": 1700,
                "constructor_elo": 1600,
                "teammate_residual": 1.2,
                "constructor_mech_rate": 0.02,
                "driver_incident_rate": 0.01,
                "circuit_residual": -0.8,
                "days_since_last_race": 14,
                "sprint_held": 1,
                "expected_position": 1.4,
                "win_probability": 0.6,
                "podium_probability": 0.8,
                "retirement_probability": 0.1,
                "mechanical_retirement_probability": 0.05,
                "position_distribution": np.array([0.5, 0.3, 0.2]),
            },
            {
                "season": season,
                "round": rnd,
                "race_date": "2024-09-01",
                "circuit_id": "monza",
                "driver_id": "b",
                "constructor_id": "red",
                "grid_position": 3,
                "quali_position": 2,
                "quali_gap_s": 0.1234,
                "driver_elo": 1550,
                "constructor_elo": 1600,
                "teammate_residual": -1.2,
                "constructor_mech_rate": 0.02,
                "driver_incident_rate": 0.03,
                "circuit_residual": 0.1,
                "days_since_last_race": 14,
                "sprint_held": 1,
                "expected_position": 2.2,
                "win_probability": 0.4,
                "podium_probability": 0.7,
                "retirement_probability": 0.2,
                "mechanical_retirement_probability": 0.05,
                "position_distribution": np.array([0.2, 0.5, 0.3]),
            },
        ]
    )


def _tables(season: int):
    races = pd.DataFrame(
        [
            {
                "season": 2020,
                "round": 1,
                "race_name": "Italian Grand Prix",
                "circuit_id": "monza",
                "circuit_name": "Autodromo Nazionale Monza",
                "race_date": "2020-09-06",
                "race_time": "13:00:00Z",
            },
            {
                "season": season,
                "round": 1,
                "race_name": "Italian Grand Prix",
                "circuit_id": "monza",
                "circuit_name": "Autodromo Nazionale Monza",
                "race_date": f"{season}-09-01",
                "race_time": "13:00:00Z",
            },
            {
                "season": season,
                "round": 2,
                "race_name": "Italian Grand Prix",
                "circuit_id": "monza",
                "circuit_name": "Autodromo Nazionale Monza",
                "race_date": "2024-09-01",
                "race_time": "13:00:00Z",
            },
        ]
    )
    results = pd.DataFrame(
        [
            _result(2020, 1, "old_winner", "red", 2, 1, 25, "Finished"),
            _result(2020, 1, "pole_sitter", "blue", 1, 4, 12, "Finished"),
            _result(season, 1, "a", "red", 1, 1, 25, "Finished"),
            _result(season, 1, "b", "red", 2, 2, 18, "Finished"),
            _result(season, 2, "future_winner", "blue", 1, 1, 25, "LEAKED_STATUS"),
        ]
    )
    sprints = pd.DataFrame(
        [
            {"season": season, "round": 1, "driver_id": "b", "constructor_id": "red", "finish_position": 1, "status": "Finished", "status_category": "finished"},
            {"season": season, "round": 2, "driver_id": "secret_sprint", "constructor_id": "blue", "finish_position": 1, "status": "Finished", "status_category": "finished"},
            {"season": season, "round": 2, "driver_id": "a", "constructor_id": "red", "finish_position": 2, "status": "Finished", "status_category": "finished"},
        ]
    )
    features = pd.DataFrame(
        [
            {"season": 2020, "round": 1, "circuit_id": "monza", "driver_id": "pole_sitter", "quali_position": 1},
            {"season": 2020, "round": 1, "circuit_id": "monza", "driver_id": "old_winner", "quali_position": 2},
            {"season": season, "round": 1, "circuit_id": "monza", "driver_id": "a", "quali_position": 1},
        ]
    )
    drivers = pd.DataFrame(
        [
            {"driver_id": "a", "code": "ALA", "permanent_number": 7, "given_name": "Alice", "family_name": "Alpha", "nationality": "Test"},
            {"driver_id": "b", "code": "BOB", "permanent_number": 8, "given_name": "Bob", "family_name": "Beta", "nationality": "Test"},
            {"driver_id": "old_winner", "code": "OLD", "permanent_number": 9, "given_name": "Oscar", "family_name": "Oldwin", "nationality": "Test"},
            {"driver_id": "pole_sitter", "code": "POL", "permanent_number": 10, "given_name": "Paula", "family_name": "Pole", "nationality": "Test"},
            {"driver_id": "future_winner", "code": "FUT", "permanent_number": 11, "given_name": "Fiona", "family_name": "Future", "nationality": "Test"},
            {"driver_id": "secret_sprint", "code": "SEC", "permanent_number": 12, "given_name": "Sam", "family_name": "Secret", "nationality": "Test"},
        ]
    )
    constructors = pd.DataFrame(
        [
            {"constructor_id": "red", "name": "Red Team"},
            {"constructor_id": "blue", "name": "Blue Team"},
        ]
    )
    circuits = pd.DataFrame(
        [
            {"circuit_id": "monza", "circuit_name": "Autodromo Nazionale Monza", "locality": "Monza", "country": "Italy"},
        ]
    )
    return races, results, drivers, constructors, circuits, sprints, features


def _result(season, rnd, driver, team, grid, place, points, status):
    return {
        "season": season,
        "round": rnd,
        "circuit_id": "monza",
        "driver_id": driver,
        "constructor_id": team,
        "grid": grid,
        "finish_position": place,
        "points": points,
        "status": status,
        "status_category": "finished",
    }


def _brief(season: int) -> str:
    races, results, drivers, constructors, circuits, sprints, features = _tables(season)
    return render_race_brief(
        _prediction(season, 2),
        races=races,
        results=results,
        drivers=drivers,
        constructors=constructors,
        circuits=circuits,
        sprints=sprints,
        features=features,
        n_sim=100,
    )


def test_brief_names_the_forecast_and_hides_this_race_result():
    text = _brief(2024)
    assert "Italian Grand Prix" in text
    assert "Autodromo Nazionale Monza, Italy" in text
    assert "Sunday 1 September 2024, 13:00 UTC" in text
    assert "Most likely winner: Alice Alpha (Red Team), 60.0%." in text
    assert "Alice Alpha" in text and "ALA" in text
    assert "Starts 3rd after qualifying 2nd, +0.123s to pole." in text
    assert "Oscar Oldwin" in text
    assert "Paula Pole" in text
    assert "Fiona Future" not in text
    assert "LEAKED_STATUS" not in text
    assert "1st with 32 points" in text
    assert "2nd with 26 points" in text
    assert "Sprint this weekend" in text
    assert "Sam Secret" in text
    assert "Against Bob Beta this season: ahead 1, behind 0." in text


def test_sprint_after_qualifying_is_left_out_before_2023():
    text = _brief(2022)
    assert "Sam Secret" not in text
    assert "not known to this forecast" in text
    assert "1st with 25 points" in text
    assert "2nd with 21 points" in text
    assert "Fiona Future" not in text


def test_sprint_points_follow_the_season_table():
    assert sprint_points(2021, 1) == 3
    assert sprint_points(2022, 3) == 1
    assert sprint_points(2022, 4) == 0
    assert sprint_points(2024, 1) == 8
    assert sprint_points(2024, 8) == 1
    assert sprint_points(2020, 1) == 0


def test_write_race_brief_sits_beside_the_json_name(tmp_path: Path):
    races, results, drivers, constructors, circuits, sprints, features = _tables(2024)
    path = write_race_brief(
        tmp_path,
        _prediction(2024, 2),
        races=races,
        results=results,
        drivers=drivers,
        constructors=constructors,
        circuits=circuits,
        sprints=sprints,
        features=features,
    )
    assert path == tmp_path / "2024_02.txt"
    assert path.read_text(encoding="utf-8").startswith("Italian Grand Prix\n")
