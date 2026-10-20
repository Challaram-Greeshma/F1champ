from datetime import date

import numpy as np
import pandas as pd
import pytest

from f1champ.eval.walkforward import evaluation_seasons, split_season
from f1champ.models.forecast import forecast_race
from f1champ.models.ranker import fit_plackett_luce
from f1champ.models.retirement import RetirementModel
from f1champ.publish.artifacts import post_qualifying_message, race_has_qualifying_info
from f1champ.schema import (
    CANDIDATE_FEATURES,
    OUTCOME_COLUMNS,
    STAGE1_FEATURES,
    STAGE1_MODEL_COLUMNS,
    assert_model_columns,
)


def test_outcome_columns_cannot_be_model_inputs():
    assert not set(OUTCOME_COLUMNS) & set(STAGE1_FEATURES)
    assert not set(OUTCOME_COLUMNS) & set(CANDIDATE_FEATURES)
    with pytest.raises(ValueError):
        assert_model_columns(["finish_position"])


def test_predict_refuses_a_race_without_qualifying_information():
    empty = pd.DataFrame(columns=["grid_position", "quali_position", "has_grid", "has_qualifying_time"])
    assert race_has_qualifying_info(empty) is False
    message = post_qualifying_message(2026, 20)
    assert "post-qualifying" in message
    assert "2026" in message and "20" in message
    with pytest.raises(SystemExit, match="post-qualifying"):
        if not race_has_qualifying_info(empty):
            raise SystemExit(message)


def test_incomplete_or_future_seasons_are_not_evaluation_seasons():
    races = pd.DataFrame(
        {
            "season": [2014, 2014, 2015],
            "round": [1, 2, 1],
            "race_date": ["2014-03-01", "2014-03-15", "2015-03-01"],
        }
    )
    results = pd.DataFrame({"season": [2014, 2014], "round": [1, 2]})
    assert evaluation_seasons(races, results, today=date(2014, 12, 31)) == [2014]
    assert evaluation_seasons(races, results, today=date(2015, 3, 1)) == [2014]


def test_split_uses_only_earlier_seasons():
    features = pd.DataFrame({"season": [2013, 2014], "round": [1, 1], "driver_id": ["a", "a"]})
    outcomes = features.copy()
    train_f, train_o, test_f, test_o = split_season(features, outcomes, 2014)
    assert list(train_f["season"]) == [2013]
    assert list(train_o["season"]) == [2013]
    assert list(test_f["season"]) == [2014]
    assert list(test_o["season"]) == [2014]


def _aligned_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_rows = []
    outcome_rows = []
    drivers = [("a", 1700, 1), ("b", 1550, 2), ("c", 1400, 3)]
    for season in (2013, 2014):
        rounds = range(1, 7) if season == 2013 else range(1, 2)
        for rnd in rounds:
            for driver, elo, place in drivers:
                feature_rows.append(
                    {
                        "season": season,
                        "round": rnd,
                        "driver_id": driver,
                        "constructor_id": driver + "_team",
                        "grid_position": place,
                        "quali_position": place,
                        "driver_elo": elo,
                        "constructor_elo": elo,
                        "constructor_mech_rate": 0.05,
                    }
                )
                outcome_rows.append(
                    {
                        "season": season,
                        "round": rnd,
                        "driver_id": driver,
                        "finish_position": place,
                        "classified": True,
                        "retired_mechanical": False,
                        "retired_incident": False,
                        "won": place == 1,
                        "status_category": "finished",
                    }
                )
    return pd.DataFrame(feature_rows), pd.DataFrame(outcome_rows)


def test_stage1_gives_the_stronger_driver_the_higher_win_probability():
    features, outcomes = _aligned_history()
    train_f = features[features["season"] == 2013]
    train_o = outcomes[outcomes["season"] == 2013]
    model = fit_plackett_luce(train_f, train_o, list(STAGE1_MODEL_COLUMNS))
    retirement = RetirementModel(intercept=-3.0, coef=0.0, rate_mean=0.05, rate_std=1.0)
    race = features[features["season"] == 2014]
    predicted = forecast_race(
        model,
        retirement,
        0.0,
        race,
        n_sim=1500,
        rng=np.random.default_rng(0),
        zero_retirement=True,
    ).set_index("driver_id")
    assert predicted.loc["a", "win_probability"] > predicted.loc["c", "win_probability"]
    assert predicted["win_probability"].sum() == pytest.approx(1.0)
