import h5py
import numpy as np

from quantem.imaging import (
    find_valid_square_scan_crop,
    quantize_4dstem_scan_crop_uint16,
    reindex_right_angle_scan_axes_to_global,
    read_known_4dstem_drift_metadata,
    read_known_drift_metadata,
    right_angle_scan_crop_to_global_bounds,
    rotated_scan_positions,
    save_known_4dstem_drift_export,
    scan_time_drift_field,
    valid_scan_position_mask,
    write_known_4dstem_drift_metadata,
    write_known_drift_metadata,
)


def test_write_and_read_known_drift_metadata(tmp_path):
    path = tmp_path / "known_drift_master.h5"
    with h5py.File(path, "w"):
        pass

    positions = np.zeros((2, 3, 2), dtype=np.float32)
    offsets = np.ones((2, 3, 2), dtype=np.float32)
    write_known_drift_metadata(
        path,
        positions_px=positions,
        positions_offset_px=offsets,
        scan_crop=(slice(4, 6), slice(7, 10)),
        detector_shape_px=(32, 32),
        label="image_0_known_right30_drift",
        source_master="source_master.h5",
        det_bin=2,
        known_drift_total_px_down_right=(0.0, 30.0),
    )

    metadata = read_known_drift_metadata(path)

    assert metadata.label == "image_0_known_right30_drift"
    assert metadata.scan_crop_rows == (4, 6)
    assert metadata.scan_crop_cols == (7, 10)
    assert metadata.detector_shape_px == (32, 32)
    assert metadata.source_master == "source_master.h5"
    assert metadata.det_bin == 2
    assert metadata.known_drift_total_px_down_right == (0.0, 30.0)
    assert metadata.probe_positions_shape == (2, 3, 2)
    assert metadata.positions_offset_shape == (2, 3, 2)
    assert metadata.scan_axes_frame is None


def test_valid_position_mask_and_square_crop_policy():
    rows, cols = np.meshgrid(
        np.arange(6, dtype=np.float32),
        np.arange(6, dtype=np.float32),
        indexing="ij",
    )
    positions = np.stack([rows, cols - 2], axis=-1)

    valid = valid_scan_position_mask(positions, source_shape=(6, 6))
    crop = find_valid_square_scan_crop(valid, 3)

    assert crop == (slice(1, 4), slice(2, 5))
    assert valid[crop].all()


def test_explicit_4dstem_metadata_aliases_match_generic_helpers(tmp_path):
    path = tmp_path / "known_4dstem_drift_master.h5"
    with h5py.File(path, "w"):
        pass

    positions = np.zeros((2, 2, 2), dtype=np.float32)
    write_known_4dstem_drift_metadata(
        path,
        positions_px=positions,
        positions_offset_px=positions,
        scan_crop=((1, 3), (4, 6)),
        detector_shape_px=(16, 16),
        label="image_1_known_right30_drift",
        source_master="source_master.h5",
        det_bin=4,
        known_drift_total_px_down_right=(0.0, 30.0),
    )

    metadata = read_known_4dstem_drift_metadata(path)

    assert metadata.label == "image_1_known_right30_drift"
    assert metadata.scan_crop_rows == (1, 3)
    assert metadata.scan_crop_cols == (4, 6)
    assert metadata.detector_shape_px == (16, 16)
    assert metadata.det_bin == 4


def test_quantize_4dstem_scan_crop_uint16_crops_scan_axes_only():
    data = np.arange(4 * 5 * 2 * 3, dtype=np.float32).reshape(4, 5, 2, 3)
    data[1, 2, 0, 0] = -2.0
    data[2, 3, 1, 2] = 70000.0

    quantized, stats = quantize_4dstem_scan_crop_uint16(
        data,
        (slice(1, 3), slice(2, 5)),
    )

    assert quantized.shape == (2, 3, 2, 3)
    assert quantized.dtype == np.uint16
    assert stats.scan_crop_rows == (1, 3)
    assert stats.scan_crop_cols == (2, 5)
    assert stats.detector_shape_px == (2, 3)
    assert stats.clipped_below == 1
    assert stats.clipped_above == 1
    assert quantized[0, 0, 0, 0] == 0
    assert quantized[1, 1, 1, 2] == np.iinfo(np.uint16).max


def test_reindex_right_angle_scan_axes_to_global_for_90_degree():
    clean = np.arange(5 * 5 * 2, dtype=np.float32).reshape(5, 5, 2)
    raw90 = np.rot90(clean, k=1, axes=(0, 1))

    globalized = reindex_right_angle_scan_axes_to_global(raw90, 90)

    np.testing.assert_array_equal(globalized, clean)


def test_right_angle_scan_crop_to_global_bounds_for_90_degree():
    rows, cols = right_angle_scan_crop_to_global_bounds(
        (slice(1, 4), slice(2, 5)),
        source_shape=(6, 6),
        scan_direction_degrees=90,
    )

    assert rows == (2, 5)
    assert cols == (2, 5)


def test_save_known_4dstem_export_globalizes_90_scan_axes_and_offsets(tmp_path):
    path = tmp_path / "known_90_global_master.h5"
    clean = np.arange(4 * 4, dtype=np.float32).reshape(4, 4, 1, 1)
    raw90 = np.rot90(clean, k=1, axes=(0, 1))
    nominal90 = rotated_scan_positions((4, 4), 90)
    drift = scan_time_drift_field((4, 4), total_drift_px=(0.0, 1.5))
    positions = nominal90 - drift
    old_offsets = positions - rotated_scan_positions((4, 4), 0)
    saved = {}

    def fake_save(master_path, data, **kwargs):
        saved["data"] = np.asarray(data)
        saved["kwargs"] = kwargs
        with h5py.File(master_path, "w"):
            pass

    result = save_known_4dstem_drift_export(
        path,
        raw90,
        scan_crop=(slice(0, 4), slice(0, 4)),
        positions_px=positions,
        positions_offset_px=old_offsets,
        nominal_positions_px=nominal90,
        label="image_1_known_right1p5_drift",
        source_master="source_master.h5",
        det_bin=2,
        known_drift_total_px_down_right=(0.0, 1.5),
        save_func=fake_save,
        scan_shape=(4, 4),
        scan_direction_degrees=90,
    )

    assert result.stats.scan_crop_rows == (0, 4)
    assert result.stats.scan_crop_cols == (0, 4)
    np.testing.assert_array_equal(saved["data"], clean.astype(np.uint16))
    assert saved["kwargs"]["metadata"]["scan_axes_frame"] == "global"

    with h5py.File(path, "r") as f:
        group = f["entry/quantem/drift"]
        np.testing.assert_allclose(
            group["positions_offset_px"][...],
            -np.rot90(drift, k=-1, axes=(0, 1)),
            rtol=0,
            atol=1e-6,
        )
        assert group.attrs["scan_axes_frame"] == "global"
        assert float(group.attrs["scan_direction_degrees"]) == 90.0
        np.testing.assert_array_equal(group.attrs["raw_scan_crop_rows"], [0, 4])

    metadata = read_known_4dstem_drift_metadata(path)
    assert metadata.scan_axes_frame == "global"
    assert metadata.scan_direction_degrees == 90.0
    assert metadata.raw_scan_crop_rows == (0, 4)
