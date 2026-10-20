from __future__ import annotations

import argparse
import logging
import sys

from f1champ.constants import N_SIM, START_SEASON
from f1champ.data.tables import ingest, load_processed
from f1champ.eval.walkforward import predict_round, run_backtest
from f1champ.features.replay import build_feature_tables
from f1champ.paths import resolve_data_dir


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="f1champ", description="Stage 1 Formula 1 results model")
    parser.add_argument("--data-dir", default="data", help="Directory for raw, processed, and prediction files")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_parser = sub.add_parser("ingest", help="Cache Jolpica history and write processed tables")
    ingest_parser.add_argument("--start-season", type=int, default=START_SEASON)
    ingest_parser.add_argument("--end-season", type=int, default=None)
    ingest_parser.add_argument("--sleep", type=float, default=0.2)

    sub.add_parser("features", help="Build leakage-safe pre-race features and separate outcomes")

    backtest_parser = sub.add_parser("backtest", help="Walk-forward from 2014 and write validation files")
    backtest_parser.add_argument("--n-sim", type=int, default=N_SIM)

    predict_parser = sub.add_parser("predict", help="Predict one round from a post-qualifying cutoff")
    predict_parser.add_argument("--season", type=int, required=True)
    predict_parser.add_argument("--round", type=int, required=True)
    predict_parser.add_argument("--n-sim", type=int, default=N_SIM)

    args = parser.parse_args(argv)
    paths = resolve_data_dir(args.data_dir)
    paths.ensure()

    if args.command == "ingest":
        ingest(paths, start_season=args.start_season, end_season=args.end_season, sleep_s=args.sleep)
        return
    if args.command == "features":
        tables = load_processed(paths)
        features, outcomes = build_feature_tables(
            tables["races"], tables["results"], tables["qualifying"], tables["sprints"]
        )
        features.to_parquet(paths.features / "driver_race.parquet", index=False)
        outcomes.to_parquet(paths.features / "outcomes.parquet", index=False)
        logging.info("Wrote %s feature rows and %s outcomes", len(features), len(outcomes))
        return
    if args.command == "backtest":
        features, outcomes, races = _load_model_tables(paths)
        run_backtest(
            features,
            outcomes,
            races,
            paths.predictions,
            paths.validation,
            n_sim=args.n_sim,
        )
        return
    if args.command == "predict":
        features, outcomes, races = _load_model_tables(paths)
        tables = load_processed(paths)
        json_path, brief_path = predict_round(
            features,
            outcomes,
            races,
            args.season,
            args.round,
            paths.predictions,
            n_sim=args.n_sim,
            results=tables["results"],
            drivers=tables["drivers"],
            constructors=tables["constructors"],
            circuits=tables["circuits"],
            sprints=tables["sprints"],
        )
        logging.info("Wrote %s", json_path)
        logging.info("Wrote %s", brief_path)
        return
    parser.error(f"Unknown command {args.command}")


def _load_model_tables(paths):
    import pandas as pd

    feature_path = paths.features / "driver_race.parquet"
    outcome_path = paths.features / "outcomes.parquet"
    if not feature_path.exists() or not outcome_path.exists():
        raise SystemExit("Missing feature tables. Run `f1champ features` first.")
    try:
        races = load_processed(paths)["races"]
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    return pd.read_parquet(feature_path), pd.read_parquet(outcome_path), races


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(0)
