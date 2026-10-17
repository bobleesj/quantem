"""Multislice transmission and gradients retain the original propagation math."""

from types import SimpleNamespace

import pytest
import torch

from quantem.diffractive_imaging.ptychography_base import PtychographyBase


@pytest.mark.parametrize("retain_probe_history", [False, True])
@pytest.mark.parametrize("num_slices", [1, 4, 16])
def test_propagation_matches_indexed_reference(num_slices, retain_probe_history):
    """Compare complex waves and every trainable gradient with indexed slices."""
    torch.manual_seed(47)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    patches_t = torch.randn(num_slices, 3, 12, 10, device=device, dtype=torch.complex64)
    probes_t = torch.randn(2, 3, 12, 10, device=device, dtype=torch.complex64)
    propagators_t = torch.exp(1j * torch.randn(max(num_slices - 1, 0), 12, 10, device=device))

    def propagate(wave_t, propagator_t):
        return torch.fft.ifft2(torch.fft.fft2(wave_t) * propagator_t)

    context = SimpleNamespace(
        num_slices=num_slices, _propagators=propagators_t, _propagate_array=propagate
    )
    outputs = []
    for candidate in (False, True):
        object_t = patches_t.clone().requires_grad_()
        probe_t = probes_t.clone().requires_grad_()
        if candidate:
            propagated_t, overlap_t = PtychographyBase.overlap_projection(
                context, object_t, probe_t, retain_probe_history=retain_probe_history
            )
        else:
            waves = [probe_t]
            overlap_t = object_t[0] * probe_t
            for index in range(1, num_slices):
                wave_t = propagate(overlap_t, propagators_t[index - 1])
                overlap_t = object_t[index] * wave_t
                waves.append(wave_t)
            propagated_t = torch.stack(waves)
        # Include the intermediate waves: analytical-gradient callers use them.
        loss_t = overlap_t.abs().square().mean()
        if retain_probe_history:
            loss_t = loss_t + propagated_t.abs().square().mean()
        elif candidate:
            assert propagated_t is None
        gradients = torch.autograd.grad(loss_t, (object_t, probe_t))
        compared = (overlap_t, *gradients)
        outputs.append((propagated_t, *compared) if retain_probe_history else compared)
    for reference_t, candidate_t in zip(*outputs):
        torch.testing.assert_close(candidate_t, reference_t, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.complex64])
@pytest.mark.parametrize("num_slices", [1, 16])
def test_overlapping_patch_gradients_match_split_reference(dtype, num_slices):
    """Complex gathering preserves object gradients where scan patches overlap."""
    from quantem.diffractive_imaging.object_models import ObjectBase

    torch.manual_seed(91)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    object_t = torch.randn(num_slices, 24, 28, device=device, dtype=dtype)
    scan_rows = torch.tensor([1, 2, 1, 3], device=device)
    scan_cols = torch.tensor([2, 2, 4, 3], device=device)
    indices_t = (
        (scan_rows[:, None, None] + torch.arange(12, device=device)[None, :, None]) * 28
        + scan_cols[:, None, None]
        + torch.arange(10, device=device)[None, None, :]
    )
    weights_t = torch.randn(num_slices, 4, 12, 10, device=device, dtype=torch.complex64)
    results = []
    for candidate in (False, True):
        value_t = object_t.clone().requires_grad_()
        if candidate:
            patches_t = ObjectBase._get_obj_patches(None, value_t, indices_t)
        else:
            transmission_t = value_t if value_t.is_complex() else torch.exp(1j * value_t)
            flat_t = transmission_t.reshape(num_slices, -1)
            patches_t = torch.complex(flat_t.real[:, indices_t], flat_t.imag[:, indices_t])
        loss_t = (patches_t.conj() * weights_t).real.sum()
        (gradient_t,) = torch.autograd.grad(loss_t, value_t)
        results.append((patches_t, gradient_t))
    for reference_t, candidate_t in zip(*results):
        torch.testing.assert_close(candidate_t, reference_t, rtol=0, atol=0)
