"""Phase arithmetic parity with the Linux glibc scalar reference."""

import platform

import numpy as np
import pytest
import torch

from quantem.diffractive_imaging._reference_phase import _phase_factor


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.skipif(platform.libc_ver()[0] != "glibc", reason="Reference uses glibc")
def test_phase_matches_reference_scalar_rounding():
    """Preserve Fourier-shift phase bits across small and reduced arguments."""
    random = np.random.default_rng(24)
    angles = np.concatenate(
        [random.uniform(-scale, scale, 10000) for scale in (1.0, 3.0, 400.0, 1e7)]
    )
    reference = np.exp(1j * angles)
    actual = _phase_factor(torch.as_tensor(angles, device="cuda")).cpu().numpy()
    assert np.array_equal(actual.view(np.uint64), reference.view(np.uint64))
