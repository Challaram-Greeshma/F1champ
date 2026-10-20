import json
from pathlib import Path

import httpx
import pytest

from f1champ.cli import main
from f1champ.data.jolpica import (
    JolpicaClient,
    parse_lap_time,
    parse_qualifying,
    parse_results,
    parse_seasons,
    representative_quali_time,
)
from f1champ.data.tables import build_tables_from_pages, save_processed
from f1champ.paths import resolve_data_dir

FIXTURES = Path(__file__).parent / "fixtures"


def test_representative_time_uses_furthest_segment_not_the_fastest():
    # Q2 is faster than Q3. The representative lap is still the Q3 time.
    assert representative_quali_time(80.0, 79.5, 79.8) == 79.8
    assert representative_quali_time(80.0, 79.5, None) == 79.5
    assert parse_lap_time("1:19.900") == 79.9


def test_fixture_results_flags_and_same_lap_inputs():
    results_page = json.loads((FIXTURES / "results_page.json").read_text(encoding="utf-8"))
    quali_page = json.loads((FIXTURES / "qualifying_page.json").read_text(encoding="utf-8"))
    parsed = parse_results([results_page])
    assert [row["driver_id"] for row in parsed] == ["damon_hill", "jacques_villeneuve", "michael_schumacher"]
    assert parsed[1]["race_time_ms"] - parsed[0]["race_time_ms"] == 24394
    quali = parse_qualifying([quali_page])
    by_driver = {row["driver_id"]: row for row in quali}
    assert by_driver["damon_hill"]["representative_s"] == 80.8
    assert by_driver["jacques_villeneuve"]["representative_s"] == 79.9

    tables = build_tables_from_pages(
        race_pages=[results_page],
        result_pages=[results_page],
        qualifying_pages=[quali_page],
        sprint_pages=[],
        status_pages=[{"MRData": {"StatusTable": {"Status": [{"statusId": "1", "status": "Finished", "count": "1"}]}}}],
        driver_pages=[{"MRData": {"DriverTable": {"Drivers": []}}}],
        constructor_pages=[{"MRData": {"ConstructorTable": {"Constructors": []}}}],
        circuit_pages=[{"MRData": {"CircuitTable": {"Circuits": []}}}],
    )
    results = tables["results"].set_index("driver_id")
    assert bool(results.loc["damon_hill", "has_grid"]) is True
    assert bool(results.loc["damon_hill", "has_race_time"]) is True
    assert bool(results.loc["damon_hill", "has_qualifying_time"]) is True
    assert bool(results.loc["michael_schumacher", "has_grid"]) is False
    assert bool(results.loc["michael_schumacher", "has_race_time"]) is False
    assert results.loc["michael_schumacher", "status_category"] == "mechanical"
    assert bool(results.loc["damon_hill", "has_lap_data"]) is False
    assert bool(results.loc["damon_hill", "has_sprint"]) is False


def test_cli_features_and_predict_guard(tmp_path):
    results_page = json.loads((FIXTURES / "results_page.json").read_text(encoding="utf-8"))
    quali_page = json.loads((FIXTURES / "qualifying_page.json").read_text(encoding="utf-8"))
    tables = build_tables_from_pages(
        race_pages=[results_page],
        result_pages=[results_page],
        qualifying_pages=[quali_page],
        sprint_pages=[],
        status_pages=[{"MRData": {"StatusTable": {"Status": [{"statusId": "1", "status": "Finished", "count": "1"}]}}}],
        driver_pages=[{"MRData": {"DriverTable": {"Drivers": []}}}],
        constructor_pages=[{"MRData": {"ConstructorTable": {"Constructors": []}}}],
        circuit_pages=[{"MRData": {"CircuitTable": {"Circuits": []}}}],
    )
    paths = resolve_data_dir(tmp_path)
    paths.ensure()
    save_processed(paths, tables)
    main(["--data-dir", str(tmp_path), "features"])
    assert (tmp_path / "features" / "driver_race.parquet").exists()
    assert (tmp_path / "features" / "outcomes.parquet").exists()
    with pytest.raises(SystemExit, match="post-qualifying"):
        main(["--data-dir", str(tmp_path), "predict", "--season", "1996", "--round", "9"])


def test_client_paginates_and_then_uses_the_cache(tmp_path):
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        offset = int(request.url.params.get("offset", "0"))
        seasons = [{"season": "1996"}] if offset == 0 else [{"season": "1997"}]
        return httpx.Response(
            200,
            json={
                "MRData": {
                    "total": "2",
                    "limit": "1",
                    "offset": str(offset),
                    "SeasonTable": {"Seasons": seasons},
                }
            },
        )

    with JolpicaClient(tmp_path, sleep_s=0, transport=httpx.MockTransport(handler)) as client:
        assert parse_seasons(client.get_pages("seasons")) == [1996, 1997]
        assert parse_seasons(client.get_pages("seasons")) == [1996, 1997]
    assert calls["n"] == 2
