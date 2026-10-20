"""Fixed Stage 1 choices. These are not tuned on the evaluation seasons."""

MODEL_VERSION = "v1"
CUTOFF = "post_qualifying"

START_SEASON = 1996
EVAL_START_SEASON = 2014

# Car identity changes enough at these seasons that last year's constructor
# rating is shrunk harder toward the field.
REGULATION_RESET_SEASONS = frozenset({2009, 2014, 2017, 2022, 2026})

ELO_INIT = 1500.0
DRIVER_K = 20.0
CONSTRUCTOR_K = 16.0
DRIVER_OFFSEASON_DECAY = 0.10
CONSTRUCTOR_CARRY = 0.70
CONSTRUCTOR_CARRY_RESET = 0.20
COUNT_OFFSEASON_DECAY = 0.50

# Beta-style prior: one mechanical failure per eleven starts (~9%).
MECH_PRIOR_ALPHA = 1.0
MECH_PRIOR_BETA = 10.0
# Incidents are less common than a raw "retired" flag once unknown statuses
# are left out of the numerator.
INCIDENT_PRIOR_ALPHA = 1.0
INCIDENT_PRIOR_BETA = 20.0

TEAMMATE_RESIDUAL_ALPHA = 0.25
CIRCUIT_SHRINK_K = 8.0

PL_L2 = 1.0
LOGISTIC_L2 = 1.0
N_SIM = 2000
RANDOM_SEED = 0
PAGE_LIMIT = 100

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"
USER_AGENT = "f1champ/0.1 (historical results model)"

LIMITATIONS = [
    "Jolpica's published grid can include corrections applied after the start.",
    "Mechanical and incident retirements are sampled independently across drivers.",
    "Lap-level pace is not used (has_lap_data is false).",
    "Expected finishing position is the mean position in simulations where that driver is classified.",
]
