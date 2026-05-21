import numpy as np

from quantem.imaging.drift_simulation import (
    correct_scalar_image_from_positions,
    integrate_virtual_detector_image,
    raw_raster_drift_effect,
    rotated_scan_positions,
    scan_time_drift_field,
    simulate_drifted_4dstem,
)
from quantem.imaging.drift_visualization import center_crop


def test_rotated_scan_positions_90_is_counterclockwise():
    positions = rotated_scan_positions((5, 5), 90)
    # At +90 degrees, the raw image appears counterclockwise on display.
    np.testing.assert_allclose(positions[0, 0], [0, 4])
    np.testing.assert_allclose(positions[-1, -1], [4, 0])


def test_rotated_scan_positions_90_displays_back_to_image0():
    image = np.arange(25, dtype=np.float32).reshape(5, 5)
    data = image[..., None]

    simulated = simulate_drifted_4dstem(
        data,
        scan_direction_degrees=90,
        total_drift_px=(0.0, 0.0),
        device="cpu",
    )["data"][..., 0]

    np.testing.assert_array_equal(np.rot90(simulated, k=-1), image)


def test_scan_time_drift_field_is_line_constant_and_hits_total():
    drift = scan_time_drift_field((5, 4), total_drift_px=(0.25, 8.0))

    assert drift.shape == (5, 4, 2)
    np.testing.assert_allclose(drift[0, :, :], 0.0)
    np.testing.assert_allclose(drift[-1, :, 0], 0.25)
    np.testing.assert_allclose(drift[-1, :, 1], 8.0)
    np.testing.assert_allclose(drift[:, 0, :], drift[:, -1, :])


def test_simulate_drifted_4dstem_samples_scan_axes_not_detector_axes():
    clean = np.zeros((4, 4, 2, 3), dtype=np.float32)
    for row in range(4):
        for col in range(4):
            clean[row, col] = 100 * row + 10 * col + np.arange(6).reshape(2, 3)

    simulated = simulate_drifted_4dstem(
        clean,
        scan_direction_degrees=0,
        total_drift_px=(0.0, 1.0),
        device="cpu",
    )
    out = simulated["data"]

    expected = np.empty_like(clean)
    for row in range(4):
        shift = row / 3.0
        for col in range(4):
            src_col = max(col - shift, 0.0)
            c0 = int(np.floor(src_col))
            c1 = min(c0 + 1, 3)
            frac = src_col - c0
            expected[row, col] = clean[row, c0] * (1 - frac) + clean[row, c1] * frac

    np.testing.assert_allclose(out, expected, rtol=2e-7, atol=4e-5)
    assert np.all(out[0, 0] == clean[0, 0])


def test_detector_integration_commutes_with_bilinear_scan_drift():
    clean = np.zeros((5, 5, 2, 3), dtype=np.float32)
    for row in range(5):
        for col in range(5):
            clean[row, col] = 100 * row + 10 * col + np.arange(6).reshape(2, 3)
    detector_mask = np.array([[True, False, True], [False, True, False]])
    drift = scan_time_drift_field((5, 5), total_drift_px=(0.75, 1.5))

    drifted_4dstem = simulate_drifted_4dstem(
        clean,
        drift_field_px=drift,
        device="cpu",
    )["data"]
    vdf_after_4dstem_drift = integrate_virtual_detector_image(drifted_4dstem, detector_mask)

    clean_vdf = integrate_virtual_detector_image(clean, detector_mask)
    drifted_vdf = simulate_drifted_4dstem(
        clean_vdf[..., None],
        drift_field_px=drift,
        device="cpu",
    )["data"][..., 0]

    np.testing.assert_allclose(vdf_after_4dstem_drift, drifted_vdf, rtol=2e-7, atol=4e-5)


def test_simulate_drifted_4dstem_can_scan_window_inside_larger_source():
    clean = np.zeros((6, 6, 1), dtype=np.float32)
    for row in range(6):
        for col in range(6):
            clean[row, col, 0] = 100 * row + col

    simulated = simulate_drifted_4dstem(
        clean,
        scan_shape=(4, 4),
        scan_origin_px=(1.0, 1.0),
        scan_direction_degrees=0,
        total_drift_px=(0.0, 0.0),
        device="cpu",
    )["data"][..., 0]

    np.testing.assert_allclose(simulated, clean[1:5, 1:5, 0])


def test_simulated_4dstem_center_crop_keeps_scan_axes():
    clean = np.arange(512 * 512, dtype=np.float32).reshape(512, 512, 1)
    simulated = simulate_drifted_4dstem(
        clean,
        total_drift_px=(0.0, 0.0),
        device="cpu",
    )["data"]

    cropped = center_crop(simulated, 400)

    assert cropped.shape == (400, 400, 1)
    np.testing.assert_allclose(cropped[..., 0], clean[56:456, 56:456, 0], atol=1e-2)


def test_raw_raster_drift_effect_uses_display_frame_not_static_rotation():
    drift = scan_time_drift_field((4, 4), total_drift_px=(2.0, 4.0))

    np.testing.assert_allclose(raw_raster_drift_effect(drift, 0), drift)
    expected_90 = np.stack([-drift[..., 1], drift[..., 0]], axis=-1)
    np.testing.assert_allclose(raw_raster_drift_effect(drift, 90), expected_90)


def test_correct_scalar_image_from_positions_recovers_no_drift_scan_window():
    clean = np.arange(36, dtype=np.float32).reshape(6, 6)
    sim = simulate_drifted_4dstem(
        clean[..., None],
        scan_shape=(4, 4),
        scan_origin_px=(1.0, 1.0),
        total_drift_px=(0.0, 0.0),
        device="cpu",
    )

    corrected = correct_scalar_image_from_positions(
        sim["data"][..., 0],
        sim["positions"],
        output_shape=(4, 4),
        output_origin_px=(1.0, 1.0),
        device="cpu",
    )

    np.testing.assert_allclose(corrected["image"], clean[1:5, 1:5])
    np.testing.assert_allclose(corrected["weight"], 1.0)


def test_positions_offset_includes_rotation_and_shared_drift():
    drift = scan_time_drift_field((4, 4), total_drift_px=(0.0, 2.0))
    sim0 = simulate_drifted_4dstem(
        np.zeros((4, 4, 1), dtype=np.float32),
        scan_direction_degrees=0,
        drift_field_px=drift,
        device="cpu",
    )
    sim90 = simulate_drifted_4dstem(
        np.zeros((4, 4, 1), dtype=np.float32),
        scan_direction_degrees=90,
        drift_field_px=drift,
        device="cpu",
    )

    np.testing.assert_allclose(sim0["drift_field_px"], sim90["drift_field_px"])
    np.testing.assert_allclose(sim0["positions_offset_px"], -drift)
    expected_90 = rotated_scan_positions((4, 4), 90) - drift - rotated_scan_positions((4, 4), 0)
    np.testing.assert_allclose(sim90["positions_offset_px"], expected_90)
