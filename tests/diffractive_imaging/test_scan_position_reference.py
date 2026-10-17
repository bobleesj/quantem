"""Raster coordinate parity with the original tutorial's NumPy affine transform."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from quantem.diffractive_imaging._tensor_preprocessing import _scan_positions
from quantem.diffractive_imaging.ptycho_utils import AffineTransform


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("masked", [False, True])
def test_rotated_raster_matches_original_coordinates(masked, device):
    """Retain exact subpixel positions for full and selected scan rasters."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires a GPU")
    shape = (128, 128)
    scan_sampling = np.array([0.429, 0.429], dtype=np.float32)
    sampling = np.array([0.1568696, 0.1568696])
    mask = np.ones(np.prod(shape), dtype=bool)
    if masked:
        mask = np.random.default_rng(4).random(mask.size) > 0.3
    angle = np.deg2rad(93)
    row, col = np.meshgrid(
        np.arange(shape[0]) * scan_sampling[0],
        np.arange(shape[1]) * scan_sampling[1],
        indexing="ij",
    )
    positions = np.stack((row.ravel(), col.ravel()), axis=-1).astype(np.float32)[mask]
    expected = AffineTransform(angle=angle)(positions, origin=positions.mean(0))
    expected -= expected.min(0)
    expected /= sampling
    expected += 64
    model = SimpleNamespace(
        scan_positions_px=torch.empty(0, dtype=torch.float32, device=device),
        gpts=shape,
        device=device,
        scan_sampling=scan_sampling,
        positions_mask=torch.as_tensor(mask, device=device),
        com_rotation_rad=angle,
        obj_sampling=sampling,
        com_transpose=False,
    )
    actual_t = _scan_positions(model, (64, 64))
    assert torch.equal(actual_t.cpu(), torch.from_numpy(expected.astype(np.float32)))
