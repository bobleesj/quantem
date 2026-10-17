"""Detector centering preserves the original NumPy reduction and fit."""

import numpy as np
import pytest
import torch

from quantem.diffractive_imaging._tensor_preprocessing import _fit_com, _measure_com


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("detector_shape", [(128, 128), (23, 27)])
def test_detector_centers_and_constant_fit_match_numpy_bitwise(detector_shape):
    """Fit the origin after computing full-precision detector moments."""
    intensities = np.random.default_rng(12).random((4, 5, *detector_shape), dtype=np.float32)
    row, col = np.indices(detector_shape)
    totals = intensities.sum(axis=(-2, -1))
    measured = np.stack(
        [(intensities * coordinate).sum(axis=(-2, -1)) / totals for coordinate in (row, col)]
    )
    measured_t = _measure_com(torch.tensor(intensities, device="cuda"), torch.float32)
    np.testing.assert_array_equal(measured_t.cpu().numpy(), measured)
    mask = np.ones((4, 5), dtype=bool)
    mask[0, 0] = False
    fitted_t = _fit_com(measured_t, torch.tensor(mask, device="cuda"), "constant", detector_shape)
    means = np.array([np.nanmean(component[mask]) for component in measured])
    expected_fit = np.broadcast_to(means[:, None, None], measured.shape)
    np.testing.assert_array_equal(fitted_t.cpu().numpy(), expected_fit)
