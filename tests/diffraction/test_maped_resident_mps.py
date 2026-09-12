"""MPS MAPED keeps corrected ANS inputs and writes a bounded packed result."""

import h5py
import hdf5plugin
import numpy as np
import pytest
import torch

pytest.importorskip("Metal")

from quantem.diffraction import MAPEDTorch


@pytest.mark.parametrize("saved", [False, True])
def test_from_files_defaults_to_ans_and_merges_late_region(tmp_path, saved):
    shape = (65, 65, 2, 4)
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
        if saved:
            from quantem.diffraction._maped_resident import ResidentMergeSource

            shifts = torch.tensor([[0.25, -0.75]], device="mps")
            direct = ResidentMergeSource(
                [source], shifts, shifts, close_sources_before_reopen=False,
                compile_merge=False,
            )
            bounded = ResidentMergeSource(
                [source], shifts, shifts, close_sources_before_reopen=False,
                compile_merge=False, compute_region_frames=130,
            )
            # Uneven work/storage boundaries preserve every float32 value.
            try:
                expected_merge = torch.cat(list(direct.blocks()))
                actual_merge = torch.cat(list(bounded.blocks()))
                assert torch.equal(actual_merge, expected_merge)
            finally:
                direct.close()
                bounded.close()
        maped.real_space_shifts = torch.zeros((1, 2), device="mps")
        maped.diffraction_shifts = torch.zeros((1, 2), device="mps")
        result = maped.merge_datasets(
            dtype="scaled_uint16", plot_result=False,
            save_to=tmp_path / "merged_master.h5" if saved else None,
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
                atol=max(region["scale"] for region in report["regions"]),
            )
        assert source.data.is_released
        assert result.metadata["maped_merge"]["backend"] == "mps"
    finally:
        if result is not None:
            result.close()
        maped.close()


@pytest.mark.parametrize("origins", [(31, 32), [(31, 32), (30, 33)]])
def test_manual_origins_align_on_the_requested_gpu(origins):
    """Manual origins keep radial coordinate grids on the selected accelerator."""
    values = torch.arange(8 * 8 * 64 * 64, device="mps").reshape(8, 8, 64, 64)
    first = ((values * 37) % 251).float()
    second = ((values * 37 + 13) % 251).float()
    maped = MAPEDTorch.from_datasets([first, second])
    maped.device = "mps"
    maped.preprocess(plot_summary=False)
    maped.diffraction_origin(origins=origins, plot_origins=False)
    maped.diffraction_align(upsample_factor=3, plot_aligned=False)
    assert maped.diffraction_shifts.device.type == "mps"
    assert torch.isfinite(maped.diffraction_shifts).all()


def test_repeated_region_passes_preserve_float32_values(tmp_path):
    """Two-pass saving preserves earlier strips while reusing working memory."""
    from quantem.gpu import io

    from quantem.diffraction._maped_resident import ResidentMergeSource

    shape = (17, 13, 8, 8)
    indices = torch.arange(np.prod(shape), device="mps").reshape(shape)
    sources = [
        io.FourDSTEMData(((indices * 13 + index * 17) % 701).to(torch.uint16), {})
        for index in range(7)
    ]
    shifts = torch.tensor(
        [[0, 0], [-1.25, 0.6], [0.75, -1.4], [2.3, 1.1], [-2.1, -0.2], [19, 0], [-18, 1]],
        device="mps",
    )
    diffraction = torch.tensor(
        [[0, 0], [0.4, -0.7], [-0.25, 0.5], [0.1, 0.2], [1.2, -0.1], [0, 0], [0.3, -0.1]],
        device="mps",
    )
    generated = ResidentMergeSource(
        sources, shifts, diffraction, close_sources_before_reopen=False
    )
    generated.region_frames = 2 * shape[1]
    try:
        first_pass = list(generated.blocks())
        expected = torch.cat(first_pass).clone()
        # Keep every yielded strip alive through another complete pass. A reused
        # output buffer would silently overwrite these earlier scientific values.
        observed = torch.cat(list(generated.blocks()))
        assert torch.equal(torch.cat(first_pass), expected)
        assert torch.equal(observed, expected)
        assert observed.dtype == torch.float32
        assert torch.isfinite(observed).all()
        output = tmp_path / "repeated_master.h5"
        io.save(output, generated, dtype="scaled_uint16", backend="mps", verbose=False)
        with io.load(output, backend="mps", verbose=False) as loaded:
            report = loaded.metadata["precision"]
            assert report["values"] == expected.numel()
            assert report["overflow"] == report["clipped"] == 0
            actual = loaded.read().reshape_as(expected)
            torch.testing.assert_close(actual, expected, rtol=0, atol=max(region["scale"] for region in report["regions"]))
    finally:
        generated.close()
        for source in sources:
            source.close()


def test_compiled_scan_regions_preserve_counts_and_changed_shifts():
    """Large regions keep exact float32 results when alignment parameters change."""
    from quantem.gpu import io

    from quantem.diffraction._maped_resident import ResidentMergeSource

    shape = (17, 256, 8, 8)
    indices_t = torch.arange(np.prod(shape), device="mps").reshape(shape)
    sources = [
        io.FourDSTEMData(((indices_t * 13 + index * 17) % 65536).to(torch.uint16), {})
        for index in range(7)
    ]
    shifts_t = torch.tensor(
        [[0, 0], [-1.25, 0.6], [0.75, -1.4], [2.3, 1.1], [-2.1, -0.2], [1.7, 0.3], [-1.8, 1.2]],
        device="mps",
    )
    diffraction_t = shifts_t * 0.1
    try:
        for displacement in (0.0, 0.35):
            before = ResidentMergeSource(
                sources, shifts_t + displacement, diffraction_t,
                close_sources_before_reopen=False, compile_merge=False,
            )
            after = ResidentMergeSource(
                sources, shifts_t + displacement, diffraction_t,
                close_sources_before_reopen=False, compile_merge=True,
            )
            before.region_frames = after.region_frames = 2048
            try:
                for expected_t, actual_t in zip(before.blocks(), after.blocks(), strict=True):
                    assert torch.equal(actual_t, expected_t)
            finally:
                before.close()
                after.close()
    finally:
        for source in sources:
            source.close()
