import matplotlib
import pytest

# Pre-load quantem.core.io before any test imports `quantem.imaging.*`
# to side-step a circular import (Dataset → io.serialize → io → file_readers →
# Dataset) that fires when the imaging tree is the first to touch core.
import quantem.core.io  # noqa: F401

matplotlib.use("Agg")


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", default=False, help="run slow tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
