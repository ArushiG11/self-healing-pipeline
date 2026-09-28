import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "observability"))
from telemetry import setup_telemetry  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _telemetry():
    """stage_span() requires setup_telemetry() to have run first (by design -- it
    fails loudly rather than silently no-op'ing observability in a real run). Do it
    once for the whole test session here rather than in every test file.
    """
    setup_telemetry()
