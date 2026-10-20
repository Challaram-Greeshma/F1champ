# f1champ

Post-qualifying Formula 1 forecasts from historical race results. For each driver in a Grand Prix the model estimates classified finishing position, win and podium probability, and retirement probability, using only the grid, qualifying position, and ratings known before the race.

Lap-level pace, weather, pit strategy, and championship simulation are not in this version.

## Setup

Python 3.11 or newer.

```terminal
python -m pip install -e ".[dev]"
```

## Run

From the project directory:

```terminal
python -m f1champ ingest
python -m f1champ features
python -m f1champ backtest
python -m f1champ predict --season 2026 --round 1
```

`ingest` downloads Jolpica seasons from 1996 through the current year into `data/raw/jolpica/` and writes parquet tables under `data/processed/`. Cached pages are not downloaded again. Requests are paced at 0.2 seconds. Pass `--sleep` to change that.

`features` rebuilds `data/features/driver_race.parquet` and `data/features/outcomes.parquet`. Feature rows are written before that race's result updates the ratings.

`backtest` scores every fully completed season from 2014 onward and writes `data/validation/metrics.json` and `data/validation/ablation.json`. A season that still has races left is not scored.

`predict` writes `data/predictions/v1/{season}_{round}.json` and a plain-text briefing at `data/predictions/v1/{season}_{round}.txt`. It runs only after that round has a grid or a qualifying result. Weights are fit on earlier seasons. Ratings inside the current season already include the races that have been run.

The text file is the same forecast in a form you can read: most likely winner and podium, the full field, the championship entering the race, previous winners at that circuit, and a note for each driver covering the grid, recent results, teammate record, and history at the circuit. It leaves out this race's result. From 2023 the sprint is included, because that sprint is already finished before Grand Prix qualifying.

Tests do not call the network:

```powershell
python -m pytest
```

## What a prediction uses

The published model uses starting grid, qualifying position, driver Elo, and constructor Elo, plus a separate mechanical-retirement model and a single incident rate for the field. Qualifying gap, teammate form, circuit history, days between races, and sprint weekends are stored on each row and are not inputs to the published probabilities.

`expected_position` is the average finish in simulations where that driver is classified. Retirement is reported beside it. Win probabilities include the chance of not finishing. Retirements are drawn independently for each driver.

## Data

Race results, qualifying, sprint results, status codes, drivers, constructors, circuits, and the calendar are fetched from the [Jolpica-F1 API](https://github.com/jolpica/jolpica-f1). 

- API: <http://api.jolpi.ca/ergast/f1/>
- Documentation: <https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md>
- Terms of use: <https://github.com/jolpica/jolpica-f1/blob/main/TERMS.md>

This project is not affiliated with Formula 1, the FIA, or Jolpica.
