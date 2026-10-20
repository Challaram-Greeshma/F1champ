from __future__ import annotations

import logging

import pandas as pd

from f1champ.constants import N_SIM
from f1champ.eval.metrics import keep_feature, ranking_metrics
from f1champ.eval.walkforward import forecast_models_for_season, iter_races, split_season
from f1champ.models.forecast import forecast_race
from f1champ.schema import CANDIDATE_FEATURES, STAGE1_MODEL_COLUMNS

logger = logging.getLogger(__name__)


def run_ablation(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    seasons: list[int],
    stage1_log_loss: dict[int, float],
    n_sim: int = N_SIM,
) -> dict:
    """Add each candidate to the Stage 1 model alone. Do not publish the kept set.

    The keep/drop decision is computed on the same seasons being scored, so
    folding a kept feature back into those predictions would leak the choice.
    """
    candidates = {}
    for candidate in CANDIDATE_FEATURES:
        logger.info("Ablating %s", candidate)
        per_season: dict[int, float] = {}
        columns = list(STAGE1_MODEL_COLUMNS) + [candidate]
        for season in seasons:
            train_features, train_outcomes, test_features, _test_outcomes = split_season(
                features, outcomes, season
            )
            if train_features.empty or test_features.empty:
                continue
            fitted = forecast_models_for_season(
                train_features,
                train_outcomes,
                columns,
                fit_baselines=False,
            )
            pieces = []
            for season_id, rnd, race_features in iter_races(test_features):
                pieces.append(
                    forecast_race(
                        fitted.ranking,
                        fitted.retirement,
                        fitted.incident_rate,
                        race_features,
                        n_sim,
                        fitted.rng(season_id, rnd, stream=0),
                    )
                )
            if not pieces:
                continue
            predicted = pd.concat(pieces, ignore_index=True)
            scored = ranking_metrics(_attach_outcomes(predicted, outcomes))
            per_season[season] = scored["win_log_loss"]
        candidates[candidate] = keep_feature(stage1_log_loss, per_season)
    return {
        "decision_metric": "win_log_loss",
        "published_model": "stage1_core",
        "note": (
            "A candidate is kept when its mean season-level win log loss improves "
            "and it improves in a majority of evaluation seasons. Kept features are "
            "not written into v1 predictions, because this choice was measured on "
            "those same seasons."
        ),
        "candidates": candidates,
    }


def _attach_outcomes(predicted: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "season",
        "round",
        "driver_id",
        "finish_position",
        "classified",
        "won",
        "retired_mechanical",
        "status_category",
    ]
    return predicted.merge(outcomes[columns], on=["season", "round", "driver_id"], how="left")
