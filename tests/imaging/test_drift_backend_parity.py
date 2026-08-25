"""Backend and precision checks for 2D non-rigid drift kernels."""

import inspect

import pytest
import torch

from quantem.imaging.drift import DriftCorrection
from quantem.imaging.drift.core.nonrigid import _regularize_knots


def _regularized_knots(
    device: torch.device,
    *,
    row_count: int,
    trend_order: int,
    num_fast_knots: int,
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(42)
    rows = torch.arange(row_count, dtype=torch.float32)
    if row_count > 1:
        rows = (rows - rows.mean()) / rows.std()
    vander = torch.stack(
        [rows**power for power in range(trend_order + 1)], dim=1
    )
    knots = torch.randn(
        2, 2, row_count, num_fast_knots, generator=generator
    )
    previous = torch.randn(
        2, 2, row_count, num_fast_knots, generator=generator
    )
    actual = knots.to(device)
    _regularize_knots(
        None,
        actual,
        previous.to(device),
        vander.to(device),
        max_shift_px=2,
        sigma_px=0.5 if row_count < 33 else 4,
        step_size=0.8,
    )
    return actual.cpu()


@pytest.mark.parametrize("num_fast_knots", [1, 2, 3])
@pytest.mark.parametrize("trend_order", [0, 1, 2, 3])
def test_float32_regularization_is_finite_for_one_to_three_knots(
    num_fast_knots: int, trend_order: int
):
    """Every supported knot layout remains finite in float32."""
    actual = _regularized_knots(
        torch.device("cpu"),
        row_count=257,
        trend_order=trend_order,
        num_fast_knots=num_fast_knots,
    )
    assert actual.dtype == torch.float32
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("num_fast_knots", [1, 2, 3])
def test_accelerator_regularization_matches_cpu(num_fast_knots: int):
    """CUDA or MPS regularization agrees with the float32 CPU reference."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        pytest.skip("backend parity requires CUDA or MPS")

    expected = _regularized_knots(
        torch.device("cpu"),
        row_count=257,
        trend_order=3,
        num_fast_knots=num_fast_knots,
    )
    actual = _regularized_knots(
        device,
        row_count=257,
        trend_order=3,
        num_fast_knots=num_fast_knots,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


def test_underdetermined_mps_fallback_matches_cpu():
    """The short-row fallback remains numerically consistent on MPS."""
    if not torch.backends.mps.is_available():
        pytest.skip("MPS fallback requires Apple Metal")
    expected = _regularized_knots(
        torch.device("cpu"), row_count=3, trend_order=3, num_fast_knots=3
    )
    actual = _regularized_knots(
        torch.device("mps"), row_count=3, trend_order=3, num_fast_knots=3
    )
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


def test_public_workflow_signatures_remain_unchanged():
    """The staged kernel work must not preempt the later API discussion."""
    expected = {
        "preprocess": {
            "pad_fraction": 0.25,
            "number_knots": 1,
            "show_merged": False,
        },
        "align_affine": {"step": 0.01, "num_tests": 9, "refine": True},
        "align_nonrigid": {
            "backend": "pytorch",
            "optimizer_name": "adam",
            "num_iterations": 8,
        },
        "generate_corrected_image": {
            "upsample_factor": 2,
            "output_original_shape": True,
        },
    }
    for method_name, defaults in expected.items():
        parameters = inspect.signature(getattr(DriftCorrection, method_name)).parameters
        for parameter_name, default in defaults.items():
            assert parameters[parameter_name].default == default
