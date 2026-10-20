from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RaceSimulation:
    win_probability: np.ndarray
    podium_probability: np.ndarray
    expected_position: np.ndarray
    mechanical_retirement_probability: np.ndarray
    retirement_probability: np.ndarray
    position_distribution: np.ndarray


def simulate_race(
    skills: np.ndarray,
    mechanical_probability: np.ndarray,
    incident_probability: float,
    n_sim: int,
    rng: np.random.Generator,
) -> RaceSimulation:
    """Sample classified order conditional on who finishes.

    Retirements are independent. A driver's expected position is the mean
    place among simulations in which that driver is classified, so a
    retirement is not stored as a slow lap. Win probability counts only
    simulations in which the driver is the first classified finisher.
    """
    pace = np.clip(np.asarray(skills, dtype=float), -30, 30)
    p_mech = np.clip(np.asarray(mechanical_probability, dtype=float), 0.0, 1.0)
    p_incident = float(np.clip(incident_probability, 0.0, 1.0))
    n_drivers = len(pace)
    if n_drivers == 0:
        empty = np.zeros(0)
        return RaceSimulation(empty, empty, empty, empty, empty, np.zeros((0, 0)))

    survive_p = (1.0 - p_mech) * (1.0 - p_incident)
    survived = rng.random((n_sim, n_drivers)) < survive_p
    latent = np.where(survived, pace + rng.gumbel(size=(n_sim, n_drivers)), -1e9)
    # greater[s, k, j] is true when driver k finishes ahead of driver j.
    n_better = (latent[:, :, None] > latent[:, None, :]).sum(axis=1)
    conditional_position = n_better + 1
    is_win = survived & (n_better == 0)
    is_podium = survived & (n_better <= 2)

    win_counts = is_win.sum(axis=0).astype(float)
    if win_counts.sum() <= 0:
        win_probability = np.full(n_drivers, 1.0 / n_drivers)
    else:
        win_probability = win_counts / win_counts.sum()

    position_sum = np.where(survived, conditional_position, 0.0).sum(axis=0)
    position_count = survived.sum(axis=0).astype(float)
    expected = np.divide(
        position_sum,
        position_count,
        out=np.full(n_drivers, float(n_drivers)),
        where=position_count > 0,
    )

    distribution = np.zeros((n_drivers, n_drivers))
    for driver in range(n_drivers):
        mask = survived[:, driver]
        if not np.any(mask):
            distribution[driver, -1] = 1.0
            continue
        places = conditional_position[mask, driver] - 1
        counts = np.bincount(places.astype(int), minlength=n_drivers)
        distribution[driver] = counts / counts.sum()

    return RaceSimulation(
        win_probability=win_probability,
        podium_probability=is_podium.mean(axis=0),
        expected_position=expected,
        mechanical_retirement_probability=p_mech,
        retirement_probability=1.0 - survive_p,
        position_distribution=distribution,
    )
