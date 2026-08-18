"""CPU and accelerator parity for non-rigid knot regularization."""

import pytest
import torch

from quantem.imaging.drift.core.nonrigid import _regularize_knots


def _accelerator_device() -> torch.device:
    """Return the first available accelerator for the parity check."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    pytest.skip("non-rigid backend parity requires CUDA or MPS")


@pytest.mark.parametrize(
    ("row_count", "trend_order"),
    [(257, 0), (257, 1), (257, 2), (257, 3), (3, 3)],
)
def test_nonrigid_regularization_matches_cpu(row_count: int, trend_order: int):
    """Accelerated regularization must match the CPU least-squares result."""
    device = _accelerator_device()
    generator = torch.Generator().manual_seed(42)
    row_coordinates_t = torch.arange(row_count, dtype=torch.float32)
    row_coordinates_t = (
        row_coordinates_t - row_coordinates_t.mean()
    ) / row_coordinates_t.std()
    vander_t = torch.stack(
        [row_coordinates_t**power for power in range(trend_order + 1)], dim=1
    )
    knots_t = torch.randn(2, 2, row_count, 3, generator=generator)
    previous_knots_t = torch.randn(2, 2, row_count, 3, generator=generator)
    sigma_px = 0.5 if row_count < 33 else 4

    expected_knots_t = knots_t.clone()
    _regularize_knots(
        expected_knots_t,
        previous_knots_t,
        vander_t,
        max_shift_px=2,
        sigma_px=sigma_px,
        step_size=0.8,
    )

    actual_knots_t = knots_t.to(device)
    _regularize_knots(
        actual_knots_t,
        previous_knots_t.to(device),
        vander_t.to(device),
        max_shift_px=2,
        sigma_px=sigma_px,
        step_size=0.8,
    )

    assert actual_knots_t.device.type == device.type
    assert torch.isfinite(actual_knots_t).all()
    torch.testing.assert_close(
        actual_knots_t.cpu(), expected_knots_t, rtol=3e-5, atol=3e-5
    )
