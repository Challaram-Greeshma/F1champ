import pandas as pd
import pytest

from f1champ.features.replay import build_feature_tables
from f1champ.schema import FEATURE_COLUMNS, OUTCOME_COLUMNS


def _tables(n_races: int):
    races = []
    results = []
    qualifying = []
    lineups = [
        ("a", "red", 1, 1, "finished"),
        ("b", "red", 2, 2, "finished"),
        ("c", "blue", 3, 3, "finished"),
        ("d", "blue", 4, 4, "finished"),
    ]
    for rnd in range(1, n_races + 1):
        races.append(
            {
                "season": 2020,
                "round": rnd,
                "race_date": f"2020-03-{rnd:02d}",
                "circuit_id": "monza" if rnd < 3 else "spa",
            }
        )
        order = lineups if rnd != 2 else list(reversed(lineups))
        for place, (driver, team, _default_pos, grid, status) in enumerate(order, start=1):
            results.append(
                {
                    "season": 2020,
                    "round": rnd,
                    "driver_id": driver,
                    "constructor_id": team,
                    "grid": grid,
                    "finish_position": place,
                    "status_category": status,
                    "points": 0,
                    "laps": 50,
                    "race_time_ms": 1_000_000 + place * 1000,
                }
            )
            qualifying.append(
                {
                    "season": 2020,
                    "round": rnd,
                    "driver_id": driver,
                    "constructor_id": team,
                    "quali_position": grid,
                    "q1_s": None,
                    "q2_s": None,
                    "q3_s": 80.0 + grid,
                    "representative_s": 80.0 + grid,
                }
            )
    return pd.DataFrame(races), pd.DataFrame(results), pd.DataFrame(qualifying)


def test_features_do_not_change_when_later_results_are_appended():
    races, results, qualifying = _tables(3)
    early_features, _early_outcomes = build_feature_tables(races.iloc[:2], results[results["round"] <= 2], qualifying)
    full_features, full_outcomes = build_feature_tables(races, results, qualifying)

    keys = ["season", "round", "driver_id"]
    early = early_features.sort_values(keys).reset_index(drop=True)
    full_early = full_features[full_features["round"] <= 2].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(early[FEATURE_COLUMNS], full_early[FEATURE_COLUMNS])

    race_one = full_features[full_features["round"] == 1]
    assert set(race_one["driver_elo"]) == {1500}
    race_two = full_features[full_features["round"] == 2].set_index("driver_id")
    assert race_two.loc["a", "driver_elo"] > race_two.loc["d", "driver_elo"]
    assert race_two.loc["a", "teammate_residual"] > 0
    assert race_two.loc["b", "teammate_residual"] < 0
    assert race_two.loc["a", "constructor_elo"] > race_two.loc["c", "constructor_elo"]
    assert pd.isna(race_one["days_since_last_race"]).all()
    assert set(full_features.loc[full_features["round"] == 2, "days_since_last_race"]) == {1}

    winner = full_outcomes[(full_outcomes["round"] == 1) & (full_outcomes["driver_id"] == "a")].iloc[0]
    assert bool(winner["won"]) is True
    assert winner["same_lap_gap_s"] == pytest.approx(0.0)
    second = full_outcomes[(full_outcomes["round"] == 1) & (full_outcomes["driver_id"] == "b")].iloc[0]
    assert second["same_lap_gap_s"] == pytest.approx(1.0)
    assert set(full_outcomes.columns) == set(OUTCOME_COLUMNS)


def test_first_visit_circuit_residual_is_zero_and_qualifying_gap_is_to_pole():
    races, results, qualifying = _tables(1)
    features, _outcomes = build_feature_tables(races, results, qualifying)
    assert set(features["circuit_residual"]) == {0.0}
    pole = features.set_index("driver_id").loc["a"]
    assert pole["quali_gap_s"] == pytest.approx(0.0)
    assert features.set_index("driver_id").loc["b", "quali_gap_s"] == pytest.approx(1.0)
    assert bool(features["has_lap_data"].any()) is False
    assert set(features["cutoff"]) == {"post_qualifying"}
