"""Compare device-resident depth profiles with the established plotting path."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantem.core.visualization import linescan
from quantem.diffractive_imaging.ptychography_base import PtychographyBase


@pytest.mark.parametrize("transpose", [False, True])
def test_gpu_crop_and_depth_profile(transpose):
    """A smooth multislice object gives the same rotated FOV and depth profile."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU profile parity.")
    depth, row, col = np.indices((4, 41, 47))
    values = (np.exp(-((row - 20) ** 2 + (col - 23) ** 2) / 100) * (depth + 1)).astype(np.float32)
    geometry = SimpleNamespace(
        dset=SimpleNamespace(com_rotation_rad=np.deg2rad(93), com_transpose=transpose),
        obj_shape_crop=(4, 25, 27),
        _to_numpy=np.asarray,
    )
    expected = PtychographyBase._crop_rotate_obj_fov(geometry, values)
    actual_t = PtychographyBase._crop_rotate_tensor(
        geometry, torch.tensor(values, device="cuda"), None, None
    )
    torch.testing.assert_close(actual_t.cpu(), torch.from_numpy(expected), rtol=1e-5, atol=2e-6)
    positions, profile = linescan(expected, center=(12, 13), phi=20, line_len=18, linewidth=5)
    positions_t, profile_t = linescan(actual_t, center=(12, 13), phi=20, line_len=18, linewidth=5)
    assert profile_t.is_cuda
    np.testing.assert_array_equal(positions_t.cpu(), positions)
    np.testing.assert_allclose(profile_t.cpu(), profile, rtol=1e-5, atol=2e-6)


def test_gpu_illumination_support():
    """Object support morphology and Gaussian edge smoothing match the original mask."""
    from scipy import ndimage

    from quantem.diffractive_imaging._tensor_preprocessing import _object_fov_mask

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for mask parity.")
    row, col = np.indices((73, 81))
    overlap = np.exp(-((row - 31) ** 2 + (col - 43) ** 2) / 200).astype(np.float32)
    expected = ndimage.binary_closing(overlap > overlap.max() * 0.3, iterations=5)
    expected = ndimage.binary_dilation(expected, iterations=8)
    expected = ndimage.gaussian_filter(expected.astype(np.float32), sigma=2)
    actual_t = _object_fov_mask(torch.tensor(overlap, device="cuda"), 8, 2)
    np.testing.assert_allclose(actual_t.cpu(), expected, rtol=1e-6, atol=2e-7)
