"""Scientific contracts for scan-axis-only 4D-STEM drift correction."""

import numpy as np

from quantem.imaging.drift import DriftCorrection


def _paired_cubes(scan_size: int = 12):
    rows, columns = np.indices((scan_size, scan_size), dtype=np.float32)
    base = 2.0 + rows + 0.5 * columns
    factors = np.arange(1, 7, dtype=np.float32).reshape(2, 3)
    first = base[..., None, None] * factors
    second = np.rot90(base, axes=(0, 1)).copy()[..., None, None] * factors
    return first.astype(np.float32), second.astype(np.float32), factors


def _prepared_4dstem():
    first, second, factors = _paired_cubes()
    correction = DriftCorrection.from_4dstem(
        first,
        second,
        scan_direction_degrees=(0.0, 90.0),
    ).preprocess(
        pad_fraction=0.5,
        pad_value=0.0,
        number_knots=1,
        show_merged=False,
        show_images=False,
    )
    return correction, first, second, factors


def test_4dstem_correction_changes_only_scan_axes():
    """Every detector pixel follows one field without detector-axis mixing."""
    correction, first, _, factors = _prepared_4dstem()
    corrected = correction.apply_correction_to_dataset(
        first,
        image_index=0,
        chunk_size=2,
    )

    assert corrected.shape == first.shape
    assert corrected.dtype == np.float32
    reference = corrected[..., 0, 0]
    supported = reference > 1e-4
    for detector_row in range(factors.shape[0]):
        for detector_column in range(factors.shape[1]):
            np.testing.assert_allclose(
                corrected[..., detector_row, detector_column][supported],
                reference[supported] * factors[detector_row, detector_column],
                rtol=2e-5,
                atol=2e-5,
            )


def test_4dstem_products_keep_native_patterns_and_coverage():
    """Probe lookup copies one native DP and canvas output records coverage."""
    correction, first, second, _ = _prepared_4dstem()
    point = {"gold": (6.0, 6.0)}
    selected = correction.diffraction_patterns_at_points(point)

    assert selected["patterns"].shape == (2, 1, 2, 2, 3)
    assert selected["selection_mode"] == "single-probe"
    initial_index = selected["scan_indices"][0, 0, 0]
    np.testing.assert_array_equal(
        selected["patterns"][0, 0, 0],
        first[tuple(initial_index)],
    )

    virtual_0 = correction.integrate_virtual_detector(first, reduce="mean")
    virtual_1 = correction.integrate_virtual_detector(second, reduce="mean")
    canvas = correction.corrected_virtual_images(
        virtual_0,
        virtual_1,
        output_frame="canvas",
    )
    canvas_shape = tuple(int(value) for value in correction.shape[-2:])
    for key in (
        "corrected_image",
        "corrected_image_0",
        "corrected_image_1",
        "coverage_image",
        "coverage_image_0",
        "coverage_image_1",
    ):
        assert canvas[key].shape == canvas_shape
    assert np.all(canvas["coverage_image"] >= 0)


def test_paired_4dstem_result_keeps_detector_sampling():
    """Corrected pair and merge retain the native detector geometry."""
    correction, first, _, _ = _prepared_4dstem()
    result = correction.corrected_4dstem(
        chunk_size=2,
        merge=True,
        verbose=False,
    )

    assert result.corrected_4dstem_0.shape == first.shape
    assert result.corrected_4dstem_1.shape == first.shape
    assert result.corrected_4dstem.shape == first.shape
    assert result.scalar_corrected_vdf.shape == first.shape[:2]
    assert result.corrected_4dstem_0.shape[-2:] == (2, 3)
