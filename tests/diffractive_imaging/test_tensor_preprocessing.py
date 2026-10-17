"""Scientific parity for device-resident raster preprocessing."""

import numpy as np
import pytest
import torch

from quantem.core.datastructures import Dataset4dstem
from quantem.diffractive_imaging import PtychographyDatasetRaster


@pytest.mark.parametrize(
    "fit_mode,bilinear",
    [("constant", False), ("plane", False), ("parabola", False), ("no_shift", True)],
)
def test_tensor_preprocessing_matches_array(fit_mode, bilinear):
    """A drifting diffraction disk keeps its calibration and amplitudes on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for device-residency parity.")
    scan_row, scan_col, detector_row, detector_col = np.indices((5, 6, 12, 16))
    center_row = 5.1 + 0.12 * scan_row + 0.03 * scan_col**2
    center_col = 7.3 - 0.07 * scan_col + 0.02 * scan_row * scan_col
    values = (
        2
        + 100 * np.exp(-((detector_row - center_row) ** 2 + (detector_col - center_col) ** 2) / 5)
    ).astype(np.float32)
    values[0, 0] = 0
    mask = np.ones((5, 6), dtype=bool)
    mask[0, 0] = False
    metadata = dict(sampling=(0.4, 0.5, 0.08, 0.09), units=("A", "A", "A^-1", "A^-1"))
    reference = PtychographyDatasetRaster.from_dataset4dstem(
        Dataset4dstem.from_array(values, **metadata), verbose=False
    )
    values_t = torch.as_tensor(values, device="cuda")
    candidate = PtychographyDatasetRaster.from_dataset4dstem(
        Dataset4dstem.from_tensor(values_t, **metadata), verbose=False
    )
    options = dict(
        com_fit_function=fit_mode,
        force_com_rotation=23,
        force_com_transpose=False,
        bilinear=bilinear,
        positions_mask=mask,
        plot_rotation=False,
        plot_com=False,
    )
    reference.preprocess(**options)
    candidate.preprocess(**options)
    assert candidate.intensities_4d is values_t
    for actual_t, expected in (
        (candidate.com_measured, reference.com_measured),
        (candidate.com_fit, reference.com_fit),
        (candidate.centered_amplitudes, reference.centered_amplitudes),
        (candidate.scan_positions_px, reference.scan_positions_px),
    ):
        assert actual_t.is_cuda
        expected = expected.detach().numpy() if isinstance(expected, torch.Tensor) else expected
        np.testing.assert_allclose(actual_t.detach().cpu().numpy(), expected, rtol=1e-5, atol=1e-5)
    assert candidate.targets.is_cuda
    assert candidate.mean_diffraction_intensity == pytest.approx(
        reference.mean_diffraction_intensity, rel=1e-6
    )
    np.testing.assert_allclose(
        candidate.amplitudes.cpu(), reference.amplitudes, rtol=1e-6, atol=1e-6
    )
    torch.testing.assert_close(
        candidate.patch_indices.cpu(), reference.patch_indices, rtol=0, atol=0
    )


def test_rotated_raster_keeps_reference_patch_rounding():
    """Odd padding preserves the tutorial's patch choices near half-pixel positions."""
    from types import SimpleNamespace

    from quantem.diffractive_imaging._tensor_preprocessing import _scan_positions
    from quantem.diffractive_imaging.ptycho_utils import AffineTransform

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for scan-coordinate parity.")
    scan_step = np.float32(0.429)
    sampling = np.array([0.15686959774922626] * 2)
    angle = np.deg2rad(93)
    row, col = np.meshgrid(np.arange(128) * scan_step, np.arange(128) * scan_step, indexing="ij")
    reference = np.stack((row.ravel(), col.ravel()), axis=-1).astype(np.float32)
    reference = AffineTransform(angle=angle)(reference, origin=reference.mean(0))
    reference -= reference.min(0)
    reference /= sampling
    reference = (reference + 33).astype(np.float32)
    geometry = SimpleNamespace(
        scan_positions_px=torch.zeros((128 * 128, 2), device="cuda"),
        gpts=(128, 128),
        device="cuda",
        scan_sampling=(scan_step, scan_step),
        positions_mask=torch.ones(128 * 128, dtype=torch.bool, device="cuda"),
        com_rotation_rad=angle,
        com_transpose=False,
        obj_sampling=sampling,
    )
    actual = _scan_positions(geometry, (33, 33)).cpu().numpy()
    np.testing.assert_array_equal(np.round(actual), np.round(reference))
    np.testing.assert_allclose(actual, reference, rtol=1e-6, atol=4e-5)
