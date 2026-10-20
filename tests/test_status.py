import pytest

from f1champ.data.status import classify_status


@pytest.mark.parametrize(
    ("status", "category"),
    [
        ("Finished", "finished"),
        ("Lapped", "finished"),
        ("+1 Lap", "finished"),
        ("+11 Laps", "finished"),
        ("Underweight", "disqualified"),
        ("Track rod", "mechanical"),
        ("Engine", "mechanical"),
        ("Gearbox", "mechanical"),
        ("Power Unit", "mechanical"),
        ("Oil leak", "mechanical"),
        ("Accident", "incident"),
        ("Collision", "incident"),
        ("Collision damage", "incident"),
        ("Spun off", "incident"),
        ("Disqualified", "disqualified"),
        ("Excluded", "disqualified"),
        ("Did not start", "dns"),
        ("Withdrew", "dns"),
        ("Retired", "unknown"),
        ("Not classified", "unknown"),
        ("Illness", "unknown"),
        ("", "unknown"),
        ("Something novel", "unknown"),
    ],
)
def test_status_categories(status, category):
    assert classify_status(status) == category
