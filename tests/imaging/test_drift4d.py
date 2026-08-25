"""4D-STEM drift propagation keeps detector coordinates scientifically intact."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from quantem.imaging.drift import CorrectionResult, DriftCorrection


def _orthogonal_4dstem_pair(scan_size: int = 24, detector_size: int = 4):
    """Build two orthogonal scans with fixed per-detector-pixel signatures."""
    rng = np.random.default_rng(14)
    image_0 = gaussian_filter(
        rng.normal(size=(scan_size, scan_size)).astype(np.float32),
        1.2,
    )
    image_0 += np.linspace(0, 2, scan_size, dtype=np.float32)[:, None]
    image_1 = np.rot90(image_0, k=-1).copy()
    detector_offset = np.arange(
        detector_size * detector_size,
        dtype=np.float32,
    ).reshape(detector_size, detector_size)
    cube_0 = image_0[..., None, None] + detector_offset
    cube_1 = image_1[..., None, None] + detector_offset
    return cube_0, cube_1, detector_offset


def _fit_small_pair(cube_0, cube_1):
    drift = DriftCorrection.from_4dstem(
        cube_0,
        cube_1,
        scan_direction_degrees=(0.0, 90.0),
        scan_sampling=0.2,
        scan_units="nm",
        device="cpu",
    ).preprocess(
        pad_fraction=0.25,
        number_knots=1,
        show_merged=False,
        show_images=False,
        show_knots=False,
    )
    drift.align_affine(
        step=0.01,
        num_tests=3,
        refine=False,
        max_image_shift=8,
        chunk_size=1,
        show_merged=False,
        show_images=False,
        show_knots=False,
    )
    return drift


def test_virtual_detector_matches_numpy_and_torch_integer_inputs():
    """Virtual integration has the same exact integer sum on both backends."""
    data = np.arange(3 * 4 * 2 * 3, dtype=np.uint16).reshape(3, 4, 2, 3)
    mask = np.array([[True, False, True], [False, True, False]])
    expected = data[..., mask].sum(axis=-1, dtype=np.uint64).astype(np.float32)

    numpy_image = DriftCorrection.integrate_virtual_detector(
        data,
        mask,
        reduce="sum",
    )
    torch_image = DriftCorrection.integrate_virtual_detector(
        torch.from_numpy(data),
        mask,
        reduce="sum",
    )

    np.testing.assert_array_equal(numpy_image, expected)
    np.testing.assert_array_equal(torch_image, expected)


def test_corrected_4dstem_transforms_scan_axes_not_detector_axes():
    """Every detector pixel receives one shared scan transform."""
    cube_0, cube_1, detector_offset = _orthogonal_4dstem_pair()
    drift = _fit_small_pair(cube_0, cube_1)

    result = drift.corrected_4dstem(chunk_size=5, verbose=False)

    assert isinstance(result, CorrectionResult)
    assert result.corrected_4dstem_0.shape == cube_0.shape
    assert result.corrected_4dstem_1.shape == cube_1.shape
    assert result.corrected_4dstem.shape == cube_0.shape
    for corrected in (result.corrected_4dstem_0, result.corrected_4dstem_1):
        detector_difference = corrected - corrected[..., :1, :1]
        np.testing.assert_allclose(
            detector_difference,
            detector_offset - detector_offset[0, 0],
            atol=2e-5,
        )


def test_regional_patterns_average_native_detector_samples():
    """Region membership changes, but diffraction pixels are not interpolated."""
    cube_0, cube_1, _ = _orthogonal_4dstem_pair(scan_size=16)
    drift = DriftCorrection.from_4dstem(
        cube_0,
        cube_1,
        scan_direction_degrees=(0.0, 90.0),
        device="cpu",
    ).preprocess(
        number_knots=1,
        show_merged=False,
        show_images=False,
        show_knots=False,
    )
    regions = {"feature": (8.0, 8.0)}

    comparison = drift.regional_diffraction_patterns(
        regions,
        radius_px=2.0,
        stages=("initial", "corrected"),
    )

    assert comparison["patterns"].shape == (2, 1, 2, 4, 4)
    for stage_index, corrected in enumerate((False, True)):
        for scan_index, cube in enumerate((cube_0, cube_1)):
            positions = drift.probe_positions(
                scan_index,
                corrected=corrected,
                strip_padding=True,
                plot=False,
            )
            mask = (
                (positions[..., 0] - 8.0) ** 2
                + (positions[..., 1] - 8.0) ** 2
                <= 2.0**2
            )
            np.testing.assert_allclose(
                comparison["patterns"][stage_index, 0, scan_index],
                cube[mask].mean(axis=0, dtype=np.float32),
            )
            assert comparison["sample_counts"][stage_index, 0, scan_index] == mask.sum()
