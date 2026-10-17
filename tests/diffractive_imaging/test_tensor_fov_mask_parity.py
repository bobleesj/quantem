"""Illuminated object support retains the original SciPy filtering result."""

import numpy as np
import pytest
import torch
from scipy import ndimage

from quantem.diffractive_imaging._tensor_preprocessing import _object_fov_mask


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("sigma", [0.7, 2.0, 4.5])
def test_illumination_mask_matches_scipy_bitwise(sigma):
    """Smooth a closed probe illumination support without changing mask bits."""
    row, col = np.mgrid[:131, :137]
    overlap = np.exp(-((row - 51) ** 2 + (col - 63) ** 2) / 900).astype(np.float32)
    overlap[45:48, 50:53] = 0
    support = ndimage.binary_closing(overlap > overlap.max() * 0.3, iterations=5)
    support = ndimage.binary_dilation(support, iterations=3)
    expected = ndimage.gaussian_filter(support.astype(np.float32), sigma=sigma)
    actual_t = _object_fov_mask(torch.tensor(overlap, device="cuda"), dilation=3, sigma=sigma)
    np.testing.assert_array_equal(actual_t.cpu().numpy(), expected)
