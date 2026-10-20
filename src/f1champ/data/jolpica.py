from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import httpx

from f1champ.constants import JOLPICA_BASE, PAGE_LIMIT, USER_AGENT

logger = logging.getLogger(__name__)


class JolpicaError(RuntimeError):
    pass


def parse_lap_time(value: str | None) -> float | None:
    """Parse a session time like '1:29.123' into seconds."""
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.upper() in {"\\N", "NULL", "NONE"}:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 1:
            return float(parts[0])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        return None
    return None


def representative_quali_time(q1: float | None, q2: float | None, q3: float | None) -> float | None:
    """Time from the furthest qualifying segment the driver reached."""
    if q3 is not None:
        return q3
    if q2 is not None:
        return q2
    return q1


def _as_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def iter_races(pages: list[dict]) -> list[dict]:
    races: list[dict] = []
    for page in pages:
        races.extend(page.get("MRData", {}).get("RaceTable", {}).get("Races", []) or [])
    return races


def parse_seasons(pages: list[dict]) -> list[int]:
    years: list[int] = []
    for page in pages:
        for row in page.get("MRData", {}).get("SeasonTable", {}).get("Seasons", []) or []:
            year = _as_int(row.get("season"))
            if year is not None:
                years.append(year)
    return sorted(set(years))


def parse_status_table(pages: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for page in pages:
        for row in page.get("MRData", {}).get("StatusTable", {}).get("Status", []) or []:
            rows.append(
                {
                    "status_id": _as_int(row.get("statusId")),
                    "status": row.get("status") or "",
                    "count": _as_int(row.get("count")),
                }
            )
    return rows


def parse_drivers(pages: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for page in pages:
        for row in page.get("MRData", {}).get("DriverTable", {}).get("Drivers", []) or []:
            rows.append(
                {
                    "driver_id": row.get("driverId"),
                    "code": row.get("code"),
                    "permanent_number": row.get("permanentNumber"),
                    "given_name": row.get("givenName"),
                    "family_name": row.get("familyName"),
                    "date_of_birth": row.get("dateOfBirth"),
                    "nationality": row.get("nationality"),
                }
            )
    return rows


def parse_constructors(pages: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for page in pages:
        for row in page.get("MRData", {}).get("ConstructorTable", {}).get("Constructors", []) or []:
            rows.append(
                {
                    "constructor_id": row.get("constructorId"),
                    "name": row.get("name"),
                    "nationality": row.get("nationality"),
                }
            )
    return rows


def parse_circuits(pages: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for page in pages:
        for row in page.get("MRData", {}).get("CircuitTable", {}).get("Circuits", []) or []:
            location = row.get("Location") or {}
            rows.append(
                {
                    "circuit_id": row.get("circuitId"),
                    "circuit_name": row.get("circuitName"),
                    "locality": location.get("locality"),
                    "country": location.get("country"),
                }
            )
    return rows


def parse_races(pages: list[dict]) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[int, int]] = set()
    for race in iter_races(pages):
        season = _as_int(race.get("season"))
        rnd = _as_int(race.get("round"))
        if season is None or rnd is None or (season, rnd) in seen:
            continue
        seen.add((season, rnd))
        circuit = race.get("Circuit") or {}
        rows.append(
            {
                "season": season,
                "round": rnd,
                "race_name": race.get("raceName"),
                "circuit_id": circuit.get("circuitId"),
                "circuit_name": circuit.get("circuitName"),
                "race_date": race.get("date"),
                "race_time": race.get("time"),
            }
        )
    return rows


def _driver_id(row: dict) -> str | None:
    driver = row.get("Driver") or {}
    return driver.get("driverId")


def _constructor_id(row: dict) -> str | None:
    constructor = row.get("Constructor") or {}
    return constructor.get("constructorId")


def _millis(row: dict) -> float | None:
    time_obj = row.get("Time") or {}
    return _as_float(time_obj.get("millis"))


def parse_results(pages: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for race in iter_races(pages):
        season = _as_int(race.get("season"))
        rnd = _as_int(race.get("round"))
        if season is None or rnd is None:
            continue
        for result in race.get("Results") or []:
            driver_id = _driver_id(result)
            if not driver_id:
                continue
            grid = _as_int(result.get("grid"))
            millis = _millis(result)
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "race_date": race.get("date"),
                    "circuit_id": (race.get("Circuit") or {}).get("circuitId"),
                    "driver_id": driver_id,
                    "constructor_id": _constructor_id(result),
                    "car_number": result.get("number"),
                    "grid": grid,
                    "finish_position": _as_int(result.get("position")),
                    "position_text": result.get("positionText"),
                    "points": _as_float(result.get("points")),
                    "laps": _as_int(result.get("laps")),
                    "status": result.get("status") or "",
                    "race_time_ms": millis,
                }
            )
    return rows


def parse_qualifying(pages: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for race in iter_races(pages):
        season = _as_int(race.get("season"))
        rnd = _as_int(race.get("round"))
        if season is None or rnd is None:
            continue
        for result in race.get("QualifyingResults") or []:
            driver_id = _driver_id(result)
            if not driver_id:
                continue
            q1 = parse_lap_time(result.get("Q1"))
            q2 = parse_lap_time(result.get("Q2"))
            q3 = parse_lap_time(result.get("Q3"))
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "driver_id": driver_id,
                    "constructor_id": _constructor_id(result),
                    "quali_position": _as_int(result.get("position")),
                    "q1_s": q1,
                    "q2_s": q2,
                    "q3_s": q3,
                    "representative_s": representative_quali_time(q1, q2, q3),
                }
            )
    return rows


def parse_sprints(pages: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for race in iter_races(pages):
        season = _as_int(race.get("season"))
        rnd = _as_int(race.get("round"))
        if season is None or rnd is None:
            continue
        sprint_rows = race.get("SprintResults") or race.get("Results") or []
        for result in sprint_rows:
            driver_id = _driver_id(result)
            if not driver_id:
                continue
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "driver_id": driver_id,
                    "constructor_id": _constructor_id(result),
                    "finish_position": _as_int(result.get("position")),
                    "grid": _as_int(result.get("grid")),
                    "status": result.get("status") or "",
                }
            )
    return rows


class JolpicaClient:
    """Cached, paginated reader for the Ergast-compatible Jolpica API."""

    def __init__(
        self,
        cache_dir: Path,
        sleep_s: float = 0.2,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 4,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.sleep_s = sleep_s
        self.max_retries = max_retries
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = httpx.Client(
            timeout=60.0,
            transport=transport,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> JolpicaClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_pages(self, resource: str) -> list[dict]:
        pages: list[dict] = []
        offset = 0
        while True:
            page = self.get_json(resource, offset)
            pages.append(page)
            meta = page.get("MRData") or {}
            total = int(meta.get("total") or 0)
            limit = int(meta.get("limit") or PAGE_LIMIT)
            if total <= 0:
                break
            offset += limit if limit > 0 else PAGE_LIMIT
            if offset >= total:
                break
            if offset > 100000:
                raise JolpicaError(f"Pagination did not finish for {resource}")
        return pages

    def get_json(self, resource: str, offset: int) -> dict:
        path = self._cache_path(resource, offset)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        url = f"{JOLPICA_BASE}/{resource.strip('/')}.json"
        payload = self._request(url, {"limit": PAGE_LIMIT, "offset": offset})
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        temporary.replace(path)
        if self.sleep_s > 0:
            time.sleep(self.sleep_s)
        return payload

    def _cache_path(self, resource: str, offset: int) -> Path:
        safe = resource.strip("/").replace("/", "__")
        return self.cache_dir / f"{safe}__offset{offset}.json"

    def _request(self, url: str, params: dict) -> dict:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.get(url, params=params)
            except httpx.TransportError as exc:
                last_error = exc
                time.sleep(self.sleep_s * (attempt + 1))
                continue
            if response.status_code == 429 or response.status_code >= 500:
                last_error = JolpicaError(f"{response.status_code} from {url}")
                time.sleep(self.sleep_s * (attempt + 1) * 2)
                continue
            if response.status_code == 404:
                raise JolpicaError(f"404 from {url}")
            try:
                response.raise_for_status()
                return response.json()
            except (httpx.HTTPStatusError, json.JSONDecodeError) as exc:
                raise JolpicaError(f"Bad response from {url}") from exc
        raise JolpicaError(f"Failed to fetch {url}") from last_error
