"""Real-data calibration parity through the public circle-fitting API."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from quantem.core.utils.diffractive_imaging_utils import fit_probe_circle


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
def test_wse2_probe_calibration_matches_original():
    """Preserve calibration bits before initializing the tutorial reconstruction."""
    raw_path = os.environ.get("QUANTEM_PTYCHO_WSE2_RAW")
    if raw_path is None or not Path(raw_path).is_file():
        pytest.skip("Set QUANTEM_PTYCHO_WSE2_RAW to the tutorial's 128x128 raw scan")
    record_dtype = np.dtype([("data", np.float32, (128, 128)), ("gap", np.uint8, 1024)])
    patterns = np.memmap(raw_path, mode="r", dtype=record_dtype, shape=(16384,))["data"]
    mean_pattern = patterns.mean(0).T.copy()
    reference = np.asarray(fit_probe_circle(mean_pattern, show=False), dtype=np.float64)
    actual = np.asarray(
        fit_probe_circle(torch.as_tensor(mean_pattern, device="cuda"), show=False),
        dtype=np.float64,
    )
    assert np.array_equal(actual.view(np.uint64), reference.view(np.uint64))
