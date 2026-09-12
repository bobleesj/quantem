"""Resident ANS tilts reproduce the established MAPED workflow."""
import os

import h5py
import numpy as np
import pytest
import torch

from quantem.diffraction import MAPEDTorch
from quantem.gpu import io
from quantem.gpu._compact.streamed import StreamedCounts
from quantem.gpu.detector import prepare

pytestmark = pytest.mark.skipif(
    os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1",
    reason="Set QUANTEM_CUDA_ANS_TEST=1 in an owned CUDA test window.",
)


def test_ans_tilts_match_dense_alignment_and_merge(tmp_path):
    cp = pytest.importorskip("cupy")
    counts = np.random.default_rng(3).integers(
        0, 100, (3, 12, 11, 8, 8), dtype="uint16"
    )
    mask = np.zeros((8, 8), dtype="uint32")
    mask[2, 3] = 1
    residents = []
    for tilt in counts:
        encoded = StreamedCounts(tilt.shape, np.uint16, mask == 0)
        encoded.append(cp.asarray(tilt.reshape(-1, 8, 8)))
        residents.append(
            io.FourDSTEMData(
                encoded,
                {
                    "working_shape": tilt.shape,
                    "pixel_mask": mask,
                    "representation": "encoded",
                },
            )
        )
    reference = counts.copy()
    reference[:, :, :, 2, 3] = 0
    expected = MAPEDTorch.from_files(
        list(range(3)), read=lambda index: torch.tensor(reference[index], device="cuda:0"),
        device="cuda:0",
    )
    actual = MAPEDTorch.from_resident(residents, device="cuda:0")
    result = None
    try:
        for model in (expected, actual):
            model.preprocess(plot_summary=False)
        for observed, wanted in zip(actual.dp_mean, expected.dp_mean, strict=True):
            torch.testing.assert_close(observed, wanted, rtol=2e-7, atol=1e-5)
        for observed, wanted in zip(actual.im_bf, expected.im_bf, strict=True):
            torch.testing.assert_close(observed, wanted, rtol=2e-7, atol=1e-5)
        real_shifts = torch.tensor(
            [[0.0, 0.0], [-1.25, 0.6], [0.75, -1.4]], device="cuda"
        )
        diffraction_shifts = torch.tensor(
            [[0.0, 0.0], [0.4, -0.7], [-0.25, 0.5]], device="cuda"
        )
        for model in (expected, actual):
            model.real_space_shifts = real_shifts
            model.diffraction_shifts = diffraction_shifts
        expected_tensor = expected.merge_datasets(
            shift_method="bilinear",
            plot_result=False,
            batch_size=3,
            accumulator_device="cuda:0",
        ).tensor
        result = actual.merge_datasets(
            dtype="scaled_uint16",
            plot_result=False,
        )
        session = prepare(result)
        report = result.metadata["precision"]
        expected_flat = expected_tensor.reshape(-1, 8, 8).cpu().numpy()
        for index in (0, 17, 12 * 11 - 1):
            np.testing.assert_allclose(
                session.frame(index),
                expected_flat[index],
                rtol=0,
                atol=max(region["scale"] for region in report["regions"]),
            )
        assert result.metadata["maped_merge"]["source_representation"] == "encoded"
        assert actual.dp_mean_merged.shape == (8, 8)
        assert actual.im_bf_merged.shape == (12, 11)
        np.testing.assert_array_equal(
            residents[0].data.decode_scan_range_device(0, 1).get()[0],
            counts[0, 0, 0],
        )
    finally:
        if result is not None:
            result.close()
        for tilt in residents:
            tilt.close()


def test_from_files_defaults_to_median_corrected_ans_on_cuda(tmp_path):
    cp = pytest.importorskip("cupy")
    counts = np.random.default_rng(8).integers(
        0, 300, (5, 6, 4, 6), dtype="uint16"
    )
    counts[..., 0, 0] = np.iinfo(np.uint16).max
    expected = counts.copy()
    expected[..., 0, 0] = np.median(
        np.stack(
            [counts[..., 0, 1], counts[..., 1, 0], counts[..., 1, 1]],
            axis=-1,
        ),
        axis=-1,
    ).astype(np.uint16)
    path = tmp_path / "tilt_master.h5"
    io.save(path, cp.asarray(counts), dtype="uint16", verbose=False, wait=True)
    with h5py.File(path, "a") as handle:
        mask = np.zeros((4, 6), np.uint8)
        mask[0, 0] = 16
        handle["entry/instrument/detector/detectorSpecific/pixel_mask"] = mask

    maped = MAPEDTorch.from_files([path], device="cuda:0")
    try:
        source = maped.datasets.sources[0]
        assert source.representation.value == "encoded"
        assert not source.lossless
        assert source.metadata["working_counts_exact"] is True
        correction = source.metadata["hot_pixel_correction"]
        assert correction["method"] == "median"
        assert correction["coordinates_row_column"] == [[0, 0]]
        assert correction["applied"] is True
        assert source.metadata["source_read_passes"] == 1
        maped.preprocess(plot_summary=False)
        np.testing.assert_allclose(
            maped.dp_mean[0].cpu().numpy(),
            expected.mean(axis=(0, 1), dtype=np.float64).astype(np.float32),
            rtol=0,
            atol=1e-5,
        )
        maped.real_space_shifts = torch.zeros((1, 2), device="cuda:0")
        maped.diffraction_shifts = torch.zeros((1, 2), device="cuda:0")
        result = maped.merge_datasets(
            save_to=tmp_path / "merged_master.h5",
            plot_result=False,
        )
        assert maped.datasets.sources == []
        assert source.data.is_released
        assert not result.metadata["maped_merge"][
            "released_sources_before_reopen"
        ]
        assert len(result.metadata["maped_merge"]["merge_generation_pass_seconds"]) == 1
    finally:
        maped.close()
    assert source.data.is_released
