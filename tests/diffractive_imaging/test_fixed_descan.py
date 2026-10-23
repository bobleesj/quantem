"""Recorded-pattern fitting keeps descan fixed without resampling the targets."""

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging.dataset_models import PtychographyDatasetRaster
from quantem.diffractive_imaging.optimize_hyperparameters import _clone_ptychography_dataset


def _dataset(center_diffraction=True, device="cpu"):
    row, col = np.indices((16, 16))
    pattern = np.exp(-((row - 8.7) ** 2 + (col - 7.4) ** 2) / 8).astype(np.float32)
    values = np.broadcast_to(pattern, (4, 5, 16, 16)).copy()
    metadata = dict(sampling=(0.4, 0.5, 0.1, 0.1), units=("A", "A", "A^-1", "A^-1"))
    source = (
        Dataset4dstem.from_array(values, **metadata)
        if device == "cpu"
        else Dataset4dstem.from_tensor(torch.as_tensor(values, device=device), **metadata)
    )
    dataset = PtychographyDatasetRaster.from_dataset4dstem(
        source, learn_descan=False, verbose=False
    )
    dataset.preprocess(
        center_diffraction=center_diffraction,
        com_fit_function="constant",
        force_com_rotation=0,
        force_com_transpose=False,
        probe_energy=200e3,
        plot_rotation=False,
        plot_com=False,
    )
    return dataset


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("target_space", ["amplitude", "intensity"])
@pytest.mark.parametrize("center_diffraction", [True, False])
def test_recorded_targets_and_fixed_ramp(device, target_space, center_diffraction):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for tensor preprocessing.")
    dataset = _dataset(center_diffraction, device)
    dataset._set_targets(target_space)
    name = "amplitudes" if target_space == "amplitude" else "intensities"
    expected = getattr(dataset, ("centered_" if center_diffraction else "") + name)
    torch.testing.assert_close(dataset.targets, expected, rtol=0, atol=0)
    assert dataset._preprocessing_params["center_diffraction"] is center_diffraction
    indices = torch.tensor([0, 7, 19], device=dataset.device)
    *_, shifts = dataset.forward(indices, (0, 0))
    if center_diffraction:
        assert shifts is None
    else:
        torch.testing.assert_close(shifts, dataset.descan_shifts[indices], rtol=0, atol=0)
        assert not shifts.requires_grad
        assert torch.count_nonzero(shifts) > 0
        assert not torch.equal(dataset.amplitudes, dataset.centered_amplitudes)


def test_hyperparameter_clone_preserves_recorded_pattern_mode():
    dataset = _dataset(center_diffraction=False)
    dataset._set_targets("amplitude")
    cloned = _clone_ptychography_dataset(dataset)
    assert cloned.center_diffraction is False
    assert cloned._preprocessing_params["center_diffraction"] is False
    cloned._set_targets("amplitude")
    torch.testing.assert_close(cloned.targets, dataset.targets, rtol=0, atol=0)
    indices = torch.tensor([0, 7])
    *_, shifts = cloned.forward(indices, (0, 0))
    torch.testing.assert_close(shifts, dataset.descan_shifts[indices], rtol=0, atol=0)
