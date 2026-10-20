from __future__ import annotations

import re

_LAP_RE = re.compile(r"^\+\d+ laps?$")

_INCIDENT = {
    "accident",
    "collision",
    "collision damage",
    "spun off",
    "fatal accident",
    "overturned",
    "damage",
    "debris",
}

_DISQUALIFIED = {
    "disqualified",
    "excluded",
    "underweight",
}

# Exact mechanical strings from the Ergast/Jolpica status list. The substring
# fallback below covers the long tail (power unit, ERS, and similar).
_MECHANICAL = {
    "engine",
    "gearbox",
    "transmission",
    "clutch",
    "hydraulics",
    "electrical",
    "radiator",
    "suspension",
    "brakes",
    "differential",
    "overheating",
    "mechanical",
    "tyre",
    "tire",
    "driver seat",
    "puncture",
    "driveshaft",
    "fuel pressure",
    "front wing",
    "water pressure",
    "refuelling",
    "wheel",
    "throttle",
    "steering",
    "technical",
    "electronics",
    "broken wing",
    "heat shield fire",
    "exhaust",
    "oil leak",
    "magneto",
    "fuel pump",
    "oil pressure",
    "engine fire",
    "engine misfire",
    "vibration",
    "wheel bearing",
    "water leak",
    "fuel leak",
    "oil line",
    "halfshaft",
    "crankshaft",
    "wheel nut",
    "axle",
    "water pump",
    "turbo",
    "battery",
    "power unit",
    "power loss",
    "brake duct",
    "rear wing",
    "fire",
    "drivetrain",
    "ignition",
    "spark plugs",
    "launch control",
    "pneumatics",
    "out of fuel",
    "fuel system",
    "cooling system",
    "ers",
    "mgu-h",
    "mgu-k",
    "hydraulics leak",
    "alternator",
    "undertray",
    "seat",
    "broken suspension",
    "cv joint",
    "track rod",
    "chassis",
    "handling",
    "stalled",
    "safety belt",
}

_MECHANICAL_HINTS = (
    "engine",
    "gearbox",
    "transmission",
    "hydraulic",
    "electrical",
    "brake",
    "tyre",
    "tire",
    "puncture",
    "suspension",
    "overheat",
    "oil",
    "fuel",
    "water",
    "exhaust",
    "clutch",
    "differential",
    "radiator",
    "battery",
    "turbo",
    "ers",
    "mgu",
    "power unit",
    "drivetrain",
    "throttle",
    "steering",
    "axle",
    "wheel",
    "wing",
    "fire",
    "leak",
    "pressure",
    "mechanical",
    "technical",
    "electronic",
    "vibration",
    "pneumat",
    "cooling",
    "spark",
    "ignition",
    "alternator",
    "crank",
    "shaft",
    "seat",
)

# "Retired" is a cause we do not know. It is not counted as mechanical and it
# does not move pace ratings.
_UNKNOWN = {
    "",
    "retired",
    "not classified",
    "safety",
    "injured",
    "injury",
    "illness",
    "eye injury",
    "driver unwell",
    "physical",
    "medical",
}


def classify_status(status: str | None) -> str:
    """Map a Jolpica status string to a Stage 1 category.

    Categories: finished, mechanical, incident, disqualified, dns, unknown.
    """
    text = " ".join(str(status or "").strip().lower().split())
    if text in {"finished", "lapped"} or _LAP_RE.fullmatch(text):
        return "finished"
    if text in _DISQUALIFIED or text.startswith("disqualified") or text.startswith("excluded"):
        return "disqualified"
    if (
        text.startswith("did not start")
        or text.startswith("withdrew")
        or text.startswith("did not qualify")
        or text.startswith("not restarted")
    ):
        return "dns"
    if text in _INCIDENT or "collision" in text or "accident" in text or text == "spun off":
        return "incident"
    if text in _UNKNOWN:
        return "unknown"
    if text in _MECHANICAL or any(hint in text for hint in _MECHANICAL_HINTS):
        return "mechanical"
    return "unknown"
