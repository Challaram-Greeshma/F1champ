from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from f1champ.constants import EVAL_START_SEASON, N_SIM, RANDOM_SEED
from f1champ.eval.metrics import ranking_metrics, retirement_metrics
from f1champ.models.baseline import (
    fit_grid_baseline,
    fit_ratings_baseline,
    grid_expected_positions,
    grid_win_probabilities,
    ratings_expected_positions,
)
from f1champ.models.forecast import forecast_race
from f1champ.models.ranker import fit_plackett_luce
from f1champ.models.retirement import fit_retirement, incident_base_rate
from f1champ.publish.artifacts import write_json, write_race_prediction
from f1champ.schema import STAGE1_MODEL_COLUMNS

logger = logging.getLogger(__name__)

OUTCOME_FIELDS = [
    "season",
    "round",
    "driver_id",
    "finish_position",
    "classified",
    "won",
    "retired_mechanical",
    "status_category",
]


@dataclass
class FittedSeason:
    ranking: object
    retirement: object
    incident_rate: float
    grid: object | None = None
    ratings: object | None = None

    @staticmethod
    def rng(season: int, rnd: int, stream: int = 0) -> np.random.Generator:
        return np.random.default_rng(RANDOM_SEED + int(season) * 100_000 + int(rnd) * 100 + int(stream))


def evaluation_seasons(
    races: pd.DataFrame,
    results: pd.DataFrame,
    start: int = EVAL_START_SEASON,
    today: date | None = None,
) -> list[int]:
    """Seasons whose every scheduled race is already in the past and has results."""
    today = today or date.today()
    if races.empty or results.empty:
        return []
    complete = []
    result_rounds: dict[int, set[int]] = {}
    for season, rounds in results.groupby("season")["round"]:
        result_rounds[int(season)] = {int(value) for value in rounds}
    today_ts = pd.Timestamp(today)
    for season, group in races.groupby("season"):
        season = int(season)
        if season < start:
            continue
        dates = pd.to_datetime(group["race_date"], errors="coerce")
        if dates.isna().any() or bool((dates.dt.normalize() > today_ts).any()):
            continue
        scheduled = {int(value) for value in group["round"]}
        if scheduled and scheduled.issubset(result_rounds.get(season, set())):
            complete.append(season)
    return sorted(complete)


def split_season(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    season: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_features = features[features["season"] < season].copy()
    train_outcomes = outcomes[outcomes["season"] < season].copy()
    test_features = features[features["season"] == season].copy()
    test_outcomes = outcomes[outcomes["season"] == season].copy()
    return train_features, train_outcomes, test_features, test_outcomes


def forecast_models_for_season(
    train_features: pd.DataFrame,
    train_outcomes: pd.DataFrame,
    feature_names: list[str],
    fit_baselines: bool,
) -> FittedSeason:
    return FittedSeason(
        ranking=fit_plackett_luce(train_features, train_outcomes, feature_names),
        retirement=fit_retirement(train_features, train_outcomes),
        incident_rate=incident_base_rate(train_features, train_outcomes),
        grid=fit_grid_baseline(train_features, train_outcomes) if fit_baselines else None,
        ratings=fit_ratings_baseline(train_features, train_outcomes) if fit_baselines else None,
    )


def iter_races(features: pd.DataFrame):
    keys = features[["season", "round"]].drop_duplicates().sort_values(["season", "round"])
    for row in keys.itertuples(index=False):
        season = int(row.season)
        rnd = int(row.round)
        race = features[(features["season"] == season) & (features["round"] == rnd)]
        if not race.empty:
            yield season, rnd, race


def run_backtest(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    races: pd.DataFrame,
    prediction_dir,
    validation_dir,
    n_sim: int = N_SIM,
    today: date | None = None,
) -> dict:
    from f1champ.eval.ablation import run_ablation

    seasons = evaluation_seasons(races, _results_from_outcomes(outcomes), today=today)
    if not seasons:
        payload = {
            "model_version": "v1",
            "evaluation_seasons": [],
            "note": "No complete seasons from 2014 onward.",
        }
        write_json(validation_dir / "metrics.json", payload)
        return payload

    collected = {name: [] for name in ("stage1", "grid_baseline", "ratings_baseline")}
    logger.info("Walk-forward seasons: %s", seasons)
    for season in seasons:
        logger.info("Fitting season %s", season)
        train_features, train_outcomes, test_features, _test_outcomes = split_season(features, outcomes, season)
        if train_features.empty or test_features.empty:
            continue
        fitted = forecast_models_for_season(
            train_features,
            train_outcomes,
            list(STAGE1_MODEL_COLUMNS),
            fit_baselines=True,
        )
        for season_id, rnd, race_features in iter_races(test_features):
            stage1 = forecast_race(
                fitted.ranking,
                fitted.retirement,
                fitted.incident_rate,
                race_features,
                n_sim,
                FittedSeason.rng(season_id, rnd, stream=0),
            )
            ratings = forecast_race(
                fitted.ratings,
                fitted.retirement,
                0.0,
                race_features,
                n_sim,
                FittedSeason.rng(season_id, rnd, stream=1),
                zero_retirement=True,
            )
            ratings["expected_position"] = ratings_expected_positions(ratings)
            grid = stage1.copy()
            grid["expected_position"] = grid_expected_positions(stage1)
            grid["win_probability"] = grid_win_probabilities(stage1, fitted.grid)
            grid["podium_probability"] = np.nan
            collected["stage1"].append(stage1)
            collected["ratings_baseline"].append(ratings)
            collected["grid_baseline"].append(grid)
            write_race_prediction(prediction_dir, stage1, n_sim=n_sim)

    frames = {name: pd.concat(parts, ignore_index=True) for name, parts in collected.items() if parts}
    if "stage1" not in frames:
        payload = {
            "model_version": "v1",
            "evaluation_seasons": seasons,
            "note": "No evaluation season had both earlier training races and races to score.",
        }
        write_json(validation_dir / "metrics.json", payload)
        return payload
    metrics = _metrics_payload(frames, outcomes, seasons, n_sim)
    write_json(validation_dir / "metrics.json", metrics)
    stage1_loss = {
        int(season): metrics["position_and_win"]["per_season"][str(season)]["stage1"]["win_log_loss"]
        for season in seasons
        if str(season) in metrics["position_and_win"]["per_season"]
    }
    ablation = run_ablation(features, outcomes, seasons, stage1_loss, n_sim=n_sim)
    write_json(validation_dir / "ablation.json", ablation)
    metrics["ablation_path"] = str(validation_dir / "ablation.json")
    return metrics


def predict_round(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    races: pd.DataFrame,
    season: int,
    rnd: int,
    prediction_dir,
    n_sim: int = N_SIM,
    results: pd.DataFrame | None = None,
    drivers: pd.DataFrame | None = None,
    constructors: pd.DataFrame | None = None,
    circuits: pd.DataFrame | None = None,
    sprints: pd.DataFrame | None = None,
):
    from f1champ.publish.artifacts import post_qualifying_message, race_has_qualifying_info
    from f1champ.publish.brief import write_race_brief

    race_features = features[(features["season"] == season) & (features["round"] == rnd)].copy()
    if not race_has_qualifying_info(race_features):
        raise SystemExit(post_qualifying_message(season, rnd))
    train_features, train_outcomes, _, _ = split_season(features, outcomes, season)
    if train_features.empty or train_outcomes.empty:
        raise SystemExit(f"No earlier seasons are available to fit a model for {season}.")
    fitted = forecast_models_for_season(
        train_features,
        train_outcomes,
        list(STAGE1_MODEL_COLUMNS),
        fit_baselines=False,
    )
    predicted = forecast_race(
        fitted.ranking,
        fitted.retirement,
        fitted.incident_rate,
        race_features,
        n_sim,
        FittedSeason.rng(season, rnd, stream=0),
    )
    if not races.empty:
        meta = races[(races["season"] == season) & (races["round"] == rnd)]
        if not meta.empty:
            predicted["race_date"] = meta["race_date"].iloc[0]
            predicted["circuit_id"] = meta["circuit_id"].iloc[0]
    json_path = write_race_prediction(prediction_dir, predicted, n_sim=n_sim)
    brief_path = write_race_brief(
        prediction_dir,
        predicted,
        races=races,
        results=results,
        drivers=drivers,
        constructors=constructors,
        circuits=circuits,
        sprints=sprints,
        features=features,
        n_sim=n_sim,
    )
    return json_path, brief_path


def _results_from_outcomes(outcomes: pd.DataFrame) -> pd.DataFrame:
    return outcomes[["season", "round"]].drop_duplicates()


def _metrics_payload(frames: dict[str, pd.DataFrame], outcomes: pd.DataFrame, seasons: list[int], n_sim: int) -> dict:
    scored = {name: _attach(frame, outcomes) for name, frame in frames.items()}
    per_season: dict[str, dict] = {}
    for season in seasons:
        block = {}
        for name, frame in scored.items():
            block[name] = ranking_metrics(frame[frame["season"] == season])
        per_season[str(season)] = block
    pooled = {name: ranking_metrics(frame) for name, frame in scored.items()}
    mean_of_seasons = {}
    for name in scored:
        mean_of_seasons[name] = _mean_season_metrics([per_season[str(season)][name] for season in seasons if str(season) in per_season])
    retirement = {}
    stage1 = scored.get("stage1", pd.DataFrame())
    if not stage1.empty:
        retirement["pooled"] = retirement_metrics(stage1)
        retirement["per_season"] = {
            str(season): retirement_metrics(stage1[stage1["season"] == season]) for season in seasons
        }
    return {
        "model_version": "v1",
        "n_sim": n_sim,
        "evaluation_seasons": seasons,
        "position_and_win": {
            "per_season": per_season,
            "pooled": pooled,
            "mean_of_seasons": mean_of_seasons,
        },
        "retirement": retirement,
        "definitions": {
            "position_mae": "Mean absolute error of expected classified position, classified finishers only.",
            "spearman": "Mean within-race Spearman correlation for races with at least two classified finishers.",
            "win_log_loss": "Mean over races of -log(winner probability).",
            "win_brier": "Mean over races of the sum of squared driver-level win errors.",
            "retirement_brier": "Mean squared error of mechanical-retirement probability over starters.",
        },
    }


def _attach(predicted: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    return predicted.merge(outcomes[OUTCOME_FIELDS], on=["season", "round", "driver_id"], how="left")


def _mean_season_metrics(rows: list[dict]) -> dict:
    summary = {}
    for key in ("position_mae", "spearman", "win_log_loss", "win_brier"):
        values = [row[key] for row in rows if row.get(key) is not None]
        summary[key] = float(np.mean(values)) if values else None
    return summary
