"""Exact public-workflow centering across the optimized batch boundary."""

import platform

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging import PtychographyDatasetRaster


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a GPU")
@pytest.mark.skipif(platform.libc_ver()[0] != "glibc", reason="Reference phase uses glibc")
@pytest.mark.parametrize("fit_mode", ["constant", "no_shift"])
def test_centering_preserves_reference_bits_across_batches(fit_mode):
    """A 513-pattern acquisition retains reference amplitudes and normalization."""
    random = np.random.default_rng(20260927)
    values = random.poisson(20, size=(9, 57, 128, 128)).astype(np.float32)
    metadata = dict(sampling=(0.4, 0.4, 0.08, 0.08), units=("A", "A", "A^-1", "A^-1"))
    reference = PtychographyDatasetRaster.from_dataset4dstem(
        Dataset4dstem.from_array(values, **metadata), verbose=False
    )
    candidate = PtychographyDatasetRaster.from_dataset4dstem(
        Dataset4dstem.from_tensor(torch.as_tensor(values, device="cuda"), **metadata),
        verbose=False,
    )
    options = dict(
        com_fit_function=fit_mode,
        force_com_rotation=0,
        force_com_transpose=False,
        plot_rotation=False,
        plot_com=False,
    )
    reference.preprocess(**options)
    candidate.preprocess(**options)
    assert candidate.centered_amplitudes.is_cuda
    for name in ("centered_amplitudes", "mean_diffraction_intensity", "mean_diffraction_amplitude"):
        expected_t = torch.as_tensor(getattr(reference, name)).detach().cpu().contiguous()
        actual_t = torch.as_tensor(getattr(candidate, name)).detach().cpu().contiguous()
        assert actual_t.dtype == expected_t.dtype, name
        assert actual_t.shape == expected_t.shape, name
        assert torch.equal(
            actual_t.reshape(-1).view(torch.uint8), expected_t.reshape(-1).view(torch.uint8)
        ), name
