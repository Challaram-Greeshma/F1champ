import pytest

from f1champ.features.ratings import DriverResult, RatingState


def _finish(driver, constructor, position):
    return DriverResult(driver, constructor, position, "finished")


def test_pairwise_elo_moves_the_winner_up_and_ignores_non_classified():
    state = RatingState()
    state.start_season_if_needed(2013)
    state.pre_race_features("a", "red", "monza")
    state.pre_race_features("b", "blue", "monza")
    state.update_from_results(
        [
            _finish("a", "red", 1),
            DriverResult("b", "blue", None, "mechanical"),
        ],
        "monza",
    )
    # One classified driver is not a pairwise race, and a mechanical DNF is not pace.
    assert state.driver_elo["a"] == pytest.approx(1500)
    assert state.driver_elo["b"] == pytest.approx(1500)
    assert state.pre_race_features("b", "blue", "monza")["constructor_mech_rate"] == pytest.approx(2 / 12)

    state.update_from_results([_finish("a", "red", 1), _finish("b", "blue", 2)], "monza")
    assert state.driver_elo["a"] > state.driver_elo["b"]
    assert state.constructor_elo["red"] > state.constructor_elo["blue"]


def test_unknown_retirement_does_not_count_as_mechanical():
    state = RatingState()
    state.start_season_if_needed(2013)
    before = state.pre_race_features("a", "red", "monza")["constructor_mech_rate"]
    state.update_from_results([DriverResult("a", "red", 12, "unknown")], "monza")
    after = state.pre_race_features("a", "red", "monza")["constructor_mech_rate"]
    assert before == pytest.approx(1 / 11)
    assert after < before


def test_regulation_reset_shrinks_constructor_elo_harder_than_a_normal_offseason():
    reset = RatingState()
    reset.start_season_if_needed(2013)
    reset.pre_race_features("a", "red", "monza")
    reset.pre_race_features("b", "blue", "monza")
    reset.constructor_elo["red"] = 1700
    reset.constructor_elo["blue"] = 1500
    reset.driver_elo["a"] = 1700
    reset.start_season_if_needed(2014)
    # Field mean is 1600. 2014 carry is 0.20, so red moves to 1620.
    assert reset.pre_race_features("a", "red", "monza")["constructor_elo"] == pytest.approx(1620)
    assert reset.driver_elo["a"] == pytest.approx(1680)

    normal = RatingState()
    normal.start_season_if_needed(2014)
    normal.pre_race_features("a", "red", "monza")
    normal.pre_race_features("b", "blue", "monza")
    normal.constructor_elo["red"] = 1700
    normal.constructor_elo["blue"] = 1500
    normal.start_season_if_needed(2015)
    assert normal.pre_race_features("a", "red", "monza")["constructor_elo"] == pytest.approx(1670)
