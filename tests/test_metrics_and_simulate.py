import math

import numpy as np
import pandas as pd
import pytest

from f1champ.eval.metrics import calibration_table, keep_feature, ranking_metrics, retirement_metrics
from f1champ.models.ranker import race_nll_and_grad
from f1champ.models.simulate import simulate_race


def test_log_loss_and_brier_match_a_known_race():
    frame = pd.DataFrame(
        {
            "season": [2020, 2020],
            "round": [1, 1],
            "driver_id": ["a", "b"],
            "won": [True, False],
            "win_probability": [0.5, 0.5],
            "expected_position": [1.0, 2.0],
            "finish_position": [1, 2],
            "classified": [True, True],
        }
    )
    metrics = ranking_metrics(frame)
    assert metrics["win_log_loss"] == pytest.approx(-math.log(0.5))
    assert metrics["win_brier"] == pytest.approx(0.5)
    assert metrics["position_mae"] == pytest.approx(0.0)
    assert metrics["spearman"] == pytest.approx(1.0)


def test_retirement_brier_and_calibration():
    frame = pd.DataFrame(
        {
            "status_category": ["finished", "mechanical"],
            "retired_mechanical": [False, True],
            "mechanical_retirement_probability": [0.5, 0.5],
        }
    )
    metrics = retirement_metrics(frame)
    assert metrics["brier"] == pytest.approx(0.25)
    assert calibration_table(
        np.array([0.0, 1.0]),
        np.array([0.5, 0.5]),
        n_bins=2,
    )[0]["count"] == 2


def test_keep_feature_requires_a_majority_of_seasons():
    kept = keep_feature({2014: 1.0, 2015: 1.0, 2016: 1.0}, {2014: 0.8, 2015: 0.8, 2016: 1.2})
    assert kept["kept"] is True
    rejected = keep_feature({2014: 1.0, 2015: 1.0}, {2014: 0.9, 2015: 1.2})
    assert rejected["kept"] is False


def test_equal_strength_nll_is_log_of_permutations():
    nll, _grad = race_nll_and_grad(np.array([0.0, 0.0]))
    assert nll == pytest.approx(math.log(2))
    nll, _grad = race_nll_and_grad(np.array([0.0, 0.0, 0.0]))
    assert nll == pytest.approx(math.log(6))


def test_pl_gradient_matches_finite_differences():
    skills = np.array([1.0, 0.2, -0.5])
    nll, grad = race_nll_and_grad(skills)
    epsilon = 1e-6
    for index in range(len(skills)):
        shifted = skills.copy()
        shifted[index] += epsilon
        nll_shifted, _grad = race_nll_and_grad(shifted)
        assert (nll_shifted - nll) / epsilon == pytest.approx(grad[index], abs=1e-4)


def test_simulated_win_probabilities_sum_to_one_and_follow_pace():
    rng = np.random.default_rng(0)
    even = simulate_race(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.0, 4000, rng)
    assert even.win_probability.sum() == pytest.approx(1.0)
    assert even.win_probability[0] == pytest.approx(0.5, abs=0.05)
    assert even.position_distribution[0].sum() == pytest.approx(1.0)

    paced = simulate_race(np.array([3.0, 0.0]), np.array([0.0, 0.0]), 0.0, 3000, np.random.default_rng(1))
    assert paced.win_probability[0] > paced.win_probability[1]

    fragile = simulate_race(np.array([4.0, 0.0]), np.array([0.85, 0.0]), 0.0, 3000, np.random.default_rng(2))
    assert fragile.win_probability[1] > fragile.win_probability[0]
    assert fragile.retirement_probability[0] == pytest.approx(0.85)
