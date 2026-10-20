from __future__ import annotations

from dataclasses import dataclass, field

from f1champ.constants import (
    CIRCUIT_SHRINK_K,
    CONSTRUCTOR_CARRY,
    CONSTRUCTOR_CARRY_RESET,
    CONSTRUCTOR_K,
    COUNT_OFFSEASON_DECAY,
    DRIVER_K,
    DRIVER_OFFSEASON_DECAY,
    ELO_INIT,
    INCIDENT_PRIOR_ALPHA,
    INCIDENT_PRIOR_BETA,
    MECH_PRIOR_ALPHA,
    MECH_PRIOR_BETA,
    REGULATION_RESET_SEASONS,
    TEAMMATE_RESIDUAL_ALPHA,
)


@dataclass
class DriverResult:
    driver_id: str
    constructor_id: str
    finish_position: int | None
    status_category: str


@dataclass
class RatingState:
    """Pre-race ratings. Callers snapshot features before `update_from_results`."""

    current_season: int | None = None
    driver_elo: dict[str, float] = field(default_factory=dict)
    driver_residual: dict[str, float] = field(default_factory=dict)
    constructor_elo: dict[str, float] = field(default_factory=dict)
    last_season_constructor_elo: dict[str, float] = field(default_factory=dict)
    last_season_mean: float = ELO_INIT
    constructor_starts: dict[str, float] = field(default_factory=dict)
    constructor_mech: dict[str, float] = field(default_factory=dict)
    driver_starts: dict[str, float] = field(default_factory=dict)
    driver_incidents: dict[str, float] = field(default_factory=dict)
    circuit_sum: dict[tuple[str, str], float] = field(default_factory=dict)
    circuit_n: dict[tuple[str, str], float] = field(default_factory=dict)
    overall_sum: dict[str, float] = field(default_factory=dict)
    overall_n: dict[str, float] = field(default_factory=dict)

    def start_season_if_needed(self, season: int) -> None:
        if self.current_season == season:
            return
        if self.current_season is not None:
            self._roll_season()
        self.current_season = season
        self.constructor_elo = {}

    def pre_race_features(self, driver_id: str, constructor_id: str, circuit_id: str) -> dict[str, float]:
        self._ensure_driver(driver_id)
        self._ensure_constructor(constructor_id)
        return {
            "driver_elo": self.driver_elo[driver_id],
            "constructor_elo": self.constructor_elo[constructor_id],
            "teammate_residual": self.driver_residual[driver_id],
            "constructor_mech_rate": self._mech_rate(constructor_id),
            "driver_incident_rate": self._incident_rate(driver_id),
            "circuit_residual": self._circuit_residual(constructor_id, circuit_id),
        }

    def update_from_results(self, results: list[DriverResult], circuit_id: str) -> None:
        for row in results:
            self._ensure_driver(row.driver_id)
            self._ensure_constructor(row.constructor_id)
            if row.status_category == "dns":
                continue
            self.constructor_starts[row.constructor_id] = self.constructor_starts.get(row.constructor_id, 0.0) + 1.0
            self.driver_starts[row.driver_id] = self.driver_starts.get(row.driver_id, 0.0) + 1.0
            if row.status_category == "mechanical":
                self.constructor_mech[row.constructor_id] = self.constructor_mech.get(row.constructor_id, 0.0) + 1.0
            if row.status_category == "incident":
                self.driver_incidents[row.driver_id] = self.driver_incidents.get(row.driver_id, 0.0) + 1.0

        classified = [
            row
            for row in results
            if row.status_category == "finished" and row.finish_position is not None
        ]
        classified.sort(key=lambda row: row.finish_position or 0)
        self._update_driver_elo(classified)
        self._update_constructor_elo(classified)
        self._update_teammate_residual(classified)
        self._update_circuit(classified, circuit_id)

    def _roll_season(self) -> None:
        if self.constructor_elo:
            self.last_season_constructor_elo = dict(self.constructor_elo)
            self.last_season_mean = sum(self.constructor_elo.values()) / len(self.constructor_elo)
        else:
            self.last_season_constructor_elo = {}
            self.last_season_mean = ELO_INIT
        for driver_id, elo in list(self.driver_elo.items()):
            self.driver_elo[driver_id] = ELO_INIT + (elo - ELO_INIT) * (1.0 - DRIVER_OFFSEASON_DECAY)
            self.driver_residual[driver_id] = self.driver_residual.get(driver_id, 0.0) * (
                1.0 - DRIVER_OFFSEASON_DECAY
            )
        for key in list(self.constructor_starts):
            self.constructor_starts[key] *= COUNT_OFFSEASON_DECAY
            self.constructor_mech[key] = self.constructor_mech.get(key, 0.0) * COUNT_OFFSEASON_DECAY
        for key in list(self.driver_starts):
            self.driver_starts[key] *= COUNT_OFFSEASON_DECAY
            self.driver_incidents[key] = self.driver_incidents.get(key, 0.0) * COUNT_OFFSEASON_DECAY
        for key in list(self.circuit_n):
            self.circuit_n[key] *= COUNT_OFFSEASON_DECAY
            self.circuit_sum[key] = self.circuit_sum.get(key, 0.0) * COUNT_OFFSEASON_DECAY
        for key in list(self.overall_n):
            self.overall_n[key] *= COUNT_OFFSEASON_DECAY
            self.overall_sum[key] = self.overall_sum.get(key, 0.0) * COUNT_OFFSEASON_DECAY

    def _ensure_driver(self, driver_id: str) -> None:
        self.driver_elo.setdefault(driver_id, ELO_INIT)
        self.driver_residual.setdefault(driver_id, 0.0)

    def _ensure_constructor(self, constructor_id: str) -> None:
        if constructor_id in self.constructor_elo:
            return
        previous = self.last_season_constructor_elo.get(constructor_id)
        if previous is None:
            self.constructor_elo[constructor_id] = ELO_INIT
            return
        carry = (
            CONSTRUCTOR_CARRY_RESET
            if self.current_season in REGULATION_RESET_SEASONS
            else CONSTRUCTOR_CARRY
        )
        mean = self.last_season_mean
        self.constructor_elo[constructor_id] = mean + (previous - mean) * carry

    def _mech_rate(self, constructor_id: str) -> float:
        starts = self.constructor_starts.get(constructor_id, 0.0)
        mech = self.constructor_mech.get(constructor_id, 0.0)
        return (mech + MECH_PRIOR_ALPHA) / (starts + MECH_PRIOR_ALPHA + MECH_PRIOR_BETA)

    def _incident_rate(self, driver_id: str) -> float:
        starts = self.driver_starts.get(driver_id, 0.0)
        incidents = self.driver_incidents.get(driver_id, 0.0)
        return (incidents + INCIDENT_PRIOR_ALPHA) / (starts + INCIDENT_PRIOR_ALPHA + INCIDENT_PRIOR_BETA)

    def _circuit_residual(self, constructor_id: str, circuit_id: str) -> float:
        n = self.circuit_n.get((constructor_id, circuit_id), 0.0)
        overall = self.overall_n.get(constructor_id, 0.0)
        if n <= 0 or overall <= 0:
            return 0.0
        circuit_avg = self.circuit_sum[(constructor_id, circuit_id)] / n
        overall_avg = self.overall_sum[constructor_id] / overall
        return (n / (n + CIRCUIT_SHRINK_K)) * (circuit_avg - overall_avg)

    def _update_driver_elo(self, classified: list[DriverResult]) -> None:
        order = [row.driver_id for row in classified]
        _pairwise_update(self.driver_elo, order, DRIVER_K, same_group=None)

    def _update_constructor_elo(self, classified: list[DriverResult]) -> None:
        order = [row.constructor_id for row in classified]
        _pairwise_update(self.constructor_elo, order, CONSTRUCTOR_K, skip_equal_pairs=True)

    def _update_teammate_residual(self, classified: list[DriverResult]) -> None:
        by_team: dict[str, list[DriverResult]] = {}
        for row in classified:
            by_team.setdefault(row.constructor_id, []).append(row)
        for row in classified:
            teammates = [other for other in by_team[row.constructor_id] if other.driver_id != row.driver_id]
            if not teammates or row.finish_position is None:
                continue
            delta = sum((other.finish_position or 0) - row.finish_position for other in teammates) / len(teammates)
            previous = self.driver_residual.get(row.driver_id, 0.0)
            self.driver_residual[row.driver_id] = (
                (1.0 - TEAMMATE_RESIDUAL_ALPHA) * previous + TEAMMATE_RESIDUAL_ALPHA * delta
            )

    def _update_circuit(self, classified: list[DriverResult], circuit_id: str) -> None:
        for row in classified:
            if row.finish_position is None:
                continue
            key = (row.constructor_id, circuit_id)
            self.circuit_sum[key] = self.circuit_sum.get(key, 0.0) + float(row.finish_position)
            self.circuit_n[key] = self.circuit_n.get(key, 0.0) + 1.0
            self.overall_sum[row.constructor_id] = self.overall_sum.get(row.constructor_id, 0.0) + float(
                row.finish_position
            )
            self.overall_n[row.constructor_id] = self.overall_n.get(row.constructor_id, 0.0) + 1.0


def _pairwise_update(
    elos: dict[str, float],
    order: list[str],
    k: float,
    same_group: list[str] | None = None,
    skip_equal_pairs: bool = False,
) -> None:
    """Batch Elo update. `order` is best-first. Equal ids are skipped when requested."""
    n = len(order)
    if n < 2:
        return
    k_eff = k / (n - 1)
    base = {item: elos.get(item, ELO_INIT) for item in order}
    delta = {item: 0.0 for item in order}
    for i, left in enumerate(order):
        for j in range(i + 1, n):
            right = order[j]
            if skip_equal_pairs and left == right:
                continue
            if same_group is not None and same_group[i] == same_group[j]:
                continue
            expected_left = 1.0 / (1.0 + 10 ** ((base[right] - base[left]) / 400.0))
            delta[left] += k_eff * (1.0 - expected_left)
            delta[right] += k_eff * (0.0 - (1.0 - expected_left))
    # Assign once per id so two cars from one constructor accumulate a single delta.
    for item, change in delta.items():
        elos[item] = base[item] + change
