"""Reference arithmetic for centering the tutorial's 128-pixel diffraction frames."""

import numpy as np
import pytest
import torch

from quantem.diffractive_imaging._reference_fft import _fft128


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_detector_forward_transform_matches_numpy_bits():
    """Preserve each complex64 rounding boundary in the forward two-axis FFT."""
    amplitudes = np.random.default_rng(24).random((3, 128, 128), dtype=np.float32)
    actual_t = torch.tensor(amplitudes, device="cuda")
    expected = amplitudes
    for axis in (-1, -2):
        actual_t = _fft128(actual_t, dim=axis)
        expected = np.fft.fft(expected, axis=axis)
        np.testing.assert_array_equal(
            actual_t.cpu().numpy().view(np.uint32), expected.view(np.uint32)
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_shifted_inverse_transform_matches_numpy_bits():
    """Preserve the double-precision inverse after applying a subpixel shift."""
    amplitudes = np.random.default_rng(25).random((3, 128, 128), dtype=np.float32)
    frequency = np.fft.fftfreq(128)
    phase = np.exp(-2j * np.pi * (0.31 * frequency[:, None] - 0.17 * frequency[None, :]))
    expected = np.fft.fft2(amplitudes) * phase
    actual_t = torch.tensor(expected, device="cuda")
    for axis in (-1, -2):
        actual_t = _fft128(actual_t, dim=axis, inverse=True)
        expected = np.fft.ifft(expected, axis=axis)
        np.testing.assert_array_equal(
            actual_t.cpu().numpy().view(np.uint64), expected.view(np.uint64)
        )
