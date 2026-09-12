"""MPS MAPED keeps corrected ANS inputs and writes a bounded packed result."""

import h5py
import hdf5plugin
import numpy as np
import pytest
import torch

pytest.importorskip("Metal")

from quantem.diffraction import MAPEDTorch


def test_from_files_defaults_to_ans_and_merges_late_region(tmp_path):
    shape = (33, 33, 2, 4)
    values = (
        np.arange(np.prod(shape), dtype=np.uint32).reshape(shape) * 13 % 701
    ).astype(np.uint16)
    mask = np.zeros(shape[2:], np.uint32)
    mask[0, 1] = 1
    values[:, :, 0, 1] = np.uint16(65535)
    path = tmp_path / "tilt_master.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "entry/data/data_000001",
            data=values.reshape(-1, *shape[2:]),
            chunks=(1, *shape[2:]),
            **hdf5plugin.Bitshuffle(cname="lz4"),
        )
        detector = handle.require_group(
            "entry/instrument/detector/detectorSpecific"
        )
        detector.create_dataset("ntrigger", data=np.uint64(np.prod(shape[:2])))
        detector.create_dataset("pixel_mask", data=mask)

    maped = MAPEDTorch.from_files([path], device="mps", backend="mps")
    result = None
    try:
        source = maped.datasets.sources[0]
        assert source.representation.value == "encoded"
        assert source.metadata["source_read_passes"] == 1
        correction = source.metadata["hot_pixel_correction"]
        assert correction["method"] == "median"
        assert correction["coordinates_row_column"] == [[0, 1]]
        assert correction["applied"] is True
        maped.preprocess(plot_summary=False)
        expected = values.copy()
        expected[:, :, 0, 1] = np.median(
            np.stack(
                [
                    values[:, :, 0, 0],
                    values[:, :, 0, 2],
                    values[:, :, 1, 0],
                    values[:, :, 1, 1],
                    values[:, :, 1, 2],
                ],
                axis=-1,
            ),
            axis=-1,
        ).astype(np.uint16)
        np.testing.assert_allclose(
            maped.dp_mean[0].cpu().numpy(),
            expected.mean(axis=(0, 1), dtype=np.float64).astype(np.float32),
            rtol=0,
            atol=1e-5,
        )
        maped.real_space_shifts = torch.zeros((1, 2), device="mps")
        maped.diffraction_shifts = torch.zeros((1, 2), device="mps")
        result = maped.merge_datasets(
            save_to=tmp_path / "merged_master.h5", plot_result=False
        )
        expected[0] = 0
        expected[-1] = 0
        expected[:, 0] = 0
        expected[:, -1] = 0
        report = result.metadata["precision"]
        for index in (0, 1025, np.prod(shape[:2]) - 1):
            np.testing.assert_allclose(
                result.data.frame(index),
                expected.reshape(-1, *shape[2:])[index],
                rtol=0,
                atol=report["scale"],
            )
        assert source.data.is_released
        assert result.metadata["maped_merge"]["backend"] == "mps"
    finally:
        if result is not None:
            result.close()
        maped.close()
