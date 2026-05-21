import h5py
import numpy as np

from quantem.imaging import (
    find_valid_square_scan_crop,
    quantize_4dstem_scan_crop_uint16,
    read_known_4dstem_drift_metadata,
    read_known_drift_metadata,
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
