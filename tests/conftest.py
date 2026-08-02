import os
from pathlib import Path

import matplotlib
import pytest

# Pre-load quantem.core.io before any test imports `quantem.imaging.*`
# to side-step a circular import (Dataset → io.serialize → io → file_readers →
# Dataset) that fires when the imaging tree is the first to touch core.
import quantem.core.io  # noqa: F401

matplotlib.use("Agg")


@pytest.fixture(scope="session")
def drift_realdata_root() -> Path:
    """Return the permitted drift tutorial data root or skip when absent.

    Set ``QUANTEM_DRIFT_TEST_DATA`` to the directory containing the ``ws2``
    and ``srtio3_xeds`` subdirectories. An explicitly configured path is a
    promise that the data exist, so an invalid path fails instead of skipping.
    """
    if configured := os.environ.get("QUANTEM_DRIFT_TEST_DATA"):
        root = Path(configured).expanduser()
        if not root.is_dir():
            pytest.fail(
                "QUANTEM_DRIFT_TEST_DATA does not point to a directory: "
                f"{root}. Set it to the drift tutorial data directory."
            )
        return root

    root = Path(__file__).resolve().parents[1] / "data" / "drift"
    if root.is_dir():
        return root
    pytest.skip(
        "permitted WS2/XEDS data not found; set QUANTEM_DRIFT_TEST_DATA to "
        "the drift data directory or place it under data/drift"
    )


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        # Real-data drift tests are locally data-driven: they should run during
        # a normal local pytest invocation whenever their external files are
        # available, and skip themselves when the files are absent. CI
        # explicitly deselects this marker in its workflow.
        if "slow" in item.keywords and "drift_realdata" not in item.keywords:
            item.add_marker(skip_slow)
