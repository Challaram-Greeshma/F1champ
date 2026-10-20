"""Column contracts shared by the replay, the model, and the tests.

Outcome columns are not model inputs. The ranker refuses anything outside
the source and derived feature sets.
"""

FEATURE_COLUMNS = [
    "season",
    "round",
    "race_date",
    "circuit_id",
    "driver_id",
    "constructor_id",
    "cutoff",
    "grid_position",
    "quali_position",
    "quali_gap_s",
    "driver_elo",
    "constructor_elo",
    "teammate_residual",
    "constructor_mech_rate",
    "driver_incident_rate",
    "circuit_residual",
    "days_since_last_race",
    "sprint_held",
    "has_grid",
    "has_qualifying_time",
    "has_sprint",
    "has_lap_data",
]

OUTCOME_COLUMNS = [
    "season",
    "round",
    "driver_id",
    "finish_position",
    "classified",
    "retired_mechanical",
    "retired_incident",
    "same_lap_gap_s",
    "won",
    "podium",
    "status_category",
    "points",
    "laps_completed",
]

# Knowable inputs named in the Stage 1 model. grid_model is derived from
# grid_position inside the design matrix; it is not a separate data source.
STAGE1_FEATURES = [
    "grid_position",
    "quali_position",
    "driver_elo",
    "constructor_elo",
]

STAGE1_MODEL_COLUMNS = [
    "grid_model",
    "quali_position",
    "driver_elo",
    "constructor_elo",
]

CANDIDATE_FEATURES = [
    "quali_gap_s",
    "teammate_residual",
    "constructor_mech_rate",
    "circuit_residual",
    "days_since_last_race",
    "sprint_held",
]

DERIVED_FEATURES = ["grid_model", "elo_sum"]

PREDICTION_FEATURE_FIELDS = [
    "grid_position",
    "quali_position",
    "quali_gap_s",
    "driver_elo",
    "constructor_elo",
    "teammate_residual",
    "constructor_mech_rate",
    "driver_incident_rate",
    "circuit_residual",
    "days_since_last_race",
    "sprint_held",
    "has_grid",
    "has_qualifying_time",
    "has_sprint",
    "has_lap_data",
]

SOURCE_FEATURES = set(STAGE1_FEATURES) | set(CANDIDATE_FEATURES)
ALLOWED_MODEL_COLUMNS = set(STAGE1_MODEL_COLUMNS) | set(CANDIDATE_FEATURES) | set(DERIVED_FEATURES)


def assert_model_columns(names: list[str]) -> None:
    unknown = [name for name in names if name not in ALLOWED_MODEL_COLUMNS]
    if unknown:
        raise ValueError(f"Column is not an allowed pre-race feature: {unknown}")
    leaked = [name for name in names if name in OUTCOME_COLUMNS]
    if leaked:
        raise ValueError(f"Outcome column used as a feature: {leaked}")
