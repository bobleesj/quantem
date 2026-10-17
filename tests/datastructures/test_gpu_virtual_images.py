"""GPU virtual detectors preserve calibrated scientific products."""

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.core.utils.diffractive_imaging_utils import fit_probe_circle


def test_gpu_virtual_images_and_probe_calibration():
    """The same diffraction disk yields matching circle/annular images and fit."""
    pytest.importorskip("quantem.gpu")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for GPU detector parity.")
    row, col = np.indices((40, 44))
    disk = (100 * np.exp(-((row - 19.2) ** 2 + (col - 22.4) ** 2) / 50)).astype(np.float32)
    values = np.arange(1, 13, dtype=np.float32).reshape(3, 4, 1, 1) * disk
    reference = Dataset4dstem.from_array(values)
    candidate = Dataset4dstem.from_tensor(torch.tensor(values, device="cuda"))
    expected_fit = fit_probe_circle(reference.dp_mean.array, show=False)
    actual_fit = fit_probe_circle(candidate.dp_mean.tensor, show=False)
    np.testing.assert_allclose(actual_fit, expected_fit, rtol=1e-6, atol=1e-6)
    for mode, geometry in (("circle", ((19.2, 22.4), 6)), ("annular", ((19.2, 22.4), (8, 15)))):
        expected = reference.get_virtual_image(mode=mode, geometry=geometry, show=False)
        actual = candidate.get_virtual_image(mode=mode, geometry=geometry, show=False)
        assert actual.tensor.is_cuda
        np.testing.assert_allclose(actual.tensor.cpu(), expected.array, rtol=1e-6, atol=1e-5)
