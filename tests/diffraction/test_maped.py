"""MAPED on tilts loaded by quantem.gpu: one sample, three beam tilts, known displacements."""

import os
from math import floor, prod

import matplotlib.pyplot as plt
import pytest
from quantem.gpu import io
from torch import arange, backends, cuda, equal, exp, float32, meshgrid, tensor, uint16, zeros
from torch.testing import assert_close

from quantem.diffraction import MAPED
from quantem.diffraction._maped_resident import _sample_scan_rows

# Each tilt is the same sample seen with the beam moved on the detector and the
# sample moved in the scan, by whole pixels: (beam_row, beam_col), (scan_row, scan_col).
DISPLACEMENTS = [((0, 0), (0, 0)), ((2, -1), (3, 1)), ((-1, 2), (-2, -3))]


# --- complete workflow ---


def test_run_recovers_the_tilt_displacements(tmp_path):
    """run() finds each tilt's beam and scan displacement relative to the first tilt."""
    tilts = _load_tilts(tmp_path, scan_shape=(32, 32), detector_shape=(16, 16))
    maped = MAPED(tilts)
    maped.run(show=False, verbose=False)
    beam_t = tensor([beam for beam, _ in DISPLACEMENTS], dtype=float32, device=maped.device)
    scan_t = tensor([scan for _, scan in DISPLACEMENTS], dtype=float32, device=maped.device)
    # a shift undoes a displacement; worst measured error is 0.060 px
    assert_close(
        maped.diffraction_shifts - maped.diffraction_shifts[0], -beam_t, rtol=0, atol=0.07
    )
    assert_close(maped.real_space_shifts - maped.real_space_shifts[0], -scan_t, rtol=0, atol=0.07)


def test_run_matches_the_step_by_step_stages(tmp_path):
    """run() reproduces the five stage calls: identical shifts and identical float32 merge."""
    tilts = _load_tilts(tmp_path, scan_shape=(32, 32), detector_shape=(16, 16))
    one, steps = MAPED(tilts), MAPED(tilts)
    run_merged_t = one.run(show=False, verbose=False, dtype="float32").read()
    steps.find_beam_center(blur=1, show=False)
    steps.align_diffraction(border_taper=2, show=False)
    steps.align_real_space(
        iterations=20, hann_window=True, max_shift=5, show=False
    )
    # run crops to the positions every tilt covers; the stage call must say so too
    steps_merged_t = steps.merge(dtype="float32", crop=True, show=False).read()
    assert equal(one.diffraction_shifts, steps.diffraction_shifts)
    assert equal(one.real_space_shifts, steps.real_space_shifts)
    assert run_merged_t.dtype == float32
    assert equal(run_merged_t, steps_merged_t)


# --- merge ---


def test_whole_pixel_scan_copies_merge_back_to_the_sample(tmp_path):
    """Tilts that are whole-pixel scan copies of one sample merge to that sample."""
    scan_only = [((0, 0), scan) for _, scan in DISPLACEMENTS]
    tilts = _load_tilts(
        tmp_path, scan_shape=(32, 32), detector_shape=(16, 16), displacements=scan_only
    )
    maped = MAPED(tilts)
    maped.diffraction_shifts = zeros((3, 2), device=maped.device)
    maped.real_space_shifts = -tensor(
        [scan for _, scan in scan_only], dtype=float32, device=maped.device
    )
    merged_t = maped.merge(dtype="float32", show=False).read()
    # away from the scan border every tilt contributes the same counts, so their mean is
    # the sample up to float32 rounding of the three-tilt average (1.2e-4 at 400 counts)
    interior = (slice(8, 24), slice(8, 24))
    assert_close(merged_t[interior], tilts[0].read()[interior].to(float32), rtol=0, atol=2e-4)


def test_scaled_merge_and_saved_file_match_float32_regions(tmp_path):
    """Scaled storage stays within its reported error, across regions and after saving."""
    # 72 x 64 scan positions exceed one 4096-frame merge region, so two regions are merged
    tilts = _load_tilts(tmp_path, scan_shape=(72, 64), detector_shape=(8, 8))
    maped = MAPED(tilts)
    maped.run(show=False, verbose=False)
    # the second region starts at scan row 64; this patch straddles it
    regions = [(0, 4, 0, 4), (62, 66, 0, 64), (68, 72, 60, 64)]
    exact = [
        maped.merge(scan_region=region, show=False).read() for region in regions
    ]
    merged = maped.merge(dtype="scaled_uint16", show=False, verbose=False)
    max_error = merged.metadata["precision"]["max_abs_error"]
    for region, exact_t in zip(regions, exact):
        assert_close(merged.read(scan_region=region), exact_t, rtol=0, atol=max_error)
    io.save(tmp_path / "merged_master.h5", merged)
    saved = io.load(tmp_path / "merged_master.h5", verbose=False)
    for region in regions:
        assert equal(saved.read(scan_region=region), merged.read(scan_region=region))


def test_saving_while_merging_can_release_the_tilts(tmp_path):
    """save_to with release_tilts frees the tilts before reopening; the default keeps them."""
    tilts = _load_tilts(tmp_path, scan_shape=(32, 32), detector_shape=(16, 16))
    maped = MAPED(tilts)
    kept = maped.run(show=False, verbose=False).read()
    assert all(not tilt.data.is_released for tilt in tilts)
    saved = maped.merge(
        save_to=tmp_path / "merged_master.h5", crop=True, release_tilts=True, show=False,
        verbose=False,
    )
    assert all(tilt.data.is_released for tilt in tilts)
    # streaming to disk uses smaller regions, each with its own calibration
    assert_close(saved.read(), kept, rtol=0, atol=2 * saved.metadata["precision"]["max_abs_error"])


# --- summaries ---


def test_summaries_are_the_plain_means(tmp_path):
    """Mean diffraction pattern and bright-field image equal the means of the decoded tilt."""
    tilts = _load_tilts(tmp_path, scan_shape=(32, 32), detector_shape=(16, 16))
    maped = MAPED(tilts)
    for tilt, dp_mean_t, im_bf_t in zip(tilts, maped.dp_mean, maped.im_bf):
        decoded_t = tilt.read()
        assert dp_mean_t.device == decoded_t.device
        # the reference mean is taken in float64, which Apple GPUs do not have
        counts_t = decoded_t.cpu().double()
        assert equal(dp_mean_t.cpu(), counts_t.mean(dim=(0, 1)).to(float32))
        assert equal(im_bf_t.cpu(), counts_t.mean(dim=(2, 3)).to(float32))


def test_plot_titles_read_the_tilt_from_the_file_name(tmp_path):
    """A file named ``..._-17.0x_0.0y_..._master.h5`` labels its tilt ``-17.0x 0.0y``; otherwise ``Tilt i``."""
    tilts = _load_tilts(tmp_path, scan_shape=(32, 32), detector_shape=(16, 16))
    files = ["sample_-17.0x_0.0y_master.h5", "sample_8.5x_-14.72y_master.h5", "sample_master.h5"]
    maped = MAPED(tilts, files)
    labels = ["-17.0x 0.0y", "8.5x -14.72y", "Tilt 2"]
    assert maped.tilt_names == labels
    assert MAPED(tilts).tilt_names == ["Tilt 0", "Tilt 1", "Tilt 2"]
    maped.show_tilts()
    figure = plt.gcf()
    assert [axis.get_title() for axis in figure.axes] == [
        f"{label}: {product}"
        for label in labels
        for product in ["bright field", "mean diffraction pattern"]
    ]
    plt.close(figure)
    maped.find_beam_center()
    figure = plt.gcf()
    assert [axis.get_title() for axis in figure.axes if axis.get_title()] == labels
    plt.close(figure)


# --- primitives ---


@pytest.mark.parametrize("device", ["cuda:0", "mps"])
def test_scan_regions_match_ordered_taps(device):
    """Border, integer, fractional, and empty overlaps preserve full-range counts."""
    if device == "cuda:0" and not cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device == "mps" and not backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    shape = (17, 13, 4, 8)
    indices_t = arange(prod(shape), device=device).reshape(shape)
    values_t = ((indices_t * 37) % 65536).to(uint16)
    for shift in [(0, 0), (-1.25, 0.6), (0.75, -1.4), (8.25, 4.5), (30, 0)]:
        row_floor, column_floor = floor(-shift[0]), floor(-shift[1])
        row_fraction = -shift[0] - row_floor
        column_fraction = -shift[1] - column_floor
        for row0, row1, column0, column1 in [(0, 8, 0, 13), (4, 12, 3, 10)]:
            expected_t = zeros((row1 - row0, column1 - column0, *shape[2:]), device=device)
            for row_delta, row_weight in [
                (row_floor, 1 - row_fraction),
                (row_floor + 1, row_fraction),
            ]:
                for column_delta, column_weight in [
                    (column_floor, 1 - column_fraction),
                    (column_floor + 1, column_fraction),
                ]:
                    weight = row_weight * column_weight
                    if weight == 0:
                        continue
                    for row in range(row0, row1):
                        for column in range(column0, column1):
                            source_row, source_column = row + row_delta, column + column_delta
                            if 0 <= source_row < shape[0] and 0 <= source_column < shape[1]:
                                expected_t[row - row0, column - column0].add_(
                                    values_t[source_row, source_column], alpha=weight
                                )
            actual_t = _sample_scan_rows(
                values_t,
                decoded_first_row=0,
                output_first_row=row0,
                output_stop_row=row1,
                shift=shift,
                output_first_column=column0,
                output_stop_column=column1,
            )
            assert equal(actual_t, expected_t)


# --- synthetic tilt series ---


def _load_tilts(tmp_path, scan_shape, detector_shape, displacements=DISPLACEMENTS) -> list:
    """Three tilts of one sample, saved as master files and loaded encoded on the GPU.

    counts = 200 * sample * beam. The sample is two blobs in the scan, so the scan
    alignment has structure to lock onto; the beam is one disk on the detector. Each
    tilt moves both by the whole pixels in ``displacements``. CUDA cases run only in an
    owned GPU0 window, because the tests share that card with interactive work.
    """
    if cuda.is_available():
        if os.environ.get("QUANTEM_CUDA_ANS_TEST") != "1":
            pytest.skip("Set QUANTEM_CUDA_ANS_TEST=1 in an owned GPU0 test window.")
        backend = "cuda"
    elif backends.mps.is_available():
        backend = "mps"
    else:
        pytest.skip("Run this case on a CUDA or MPS accelerator.")
    scan_row, scan_col = meshgrid(
        arange(scan_shape[0], dtype=float32, device=backend),
        arange(scan_shape[1], dtype=float32, device=backend),
        indexing="ij",
    )
    k_row, k_col = meshgrid(
        arange(detector_shape[0], dtype=float32, device=backend),
        arange(detector_shape[1], dtype=float32, device=backend),
        indexing="ij",
    )
    center_row, center_col = scan_shape[0] // 2, scan_shape[1] // 2
    files = []
    for index, (beam, scan) in enumerate(displacements):
        row, col = scan_row - scan[0], scan_col - scan[1]
        sample = (
            1
            + exp(-((row - center_row) ** 2 + (col - center_col) ** 2) / 18)
            + 0.5 * exp(-((row - center_row + 8) ** 2 + (col - center_col - 6) ** 2) / 8)
        )
        disk = exp(
            -(
                (k_row - detector_shape[0] // 2 - beam[0]) ** 2
                + (k_col - detector_shape[1] // 2 - beam[1]) ** 2
            )
            / (detector_shape[0] / 2)
        )
        # detector counts are integers and the file format stores uint16
        counts_t = (200 * sample[:, :, None, None] * disk).round().to(uint16)
        path = tmp_path / f"tilt_{index}_master.h5"
        io.save(path, counts_t, dtype="uint16", backend=backend, verbose=False, wait=True)
        files.append(path)
    return io.load(files, verbose=False)
