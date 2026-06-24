"""Fast, synthetic, no-data-load unit tests for the single-GPU streaming merge.

These run in well under a second on CPU - no real 4D-STEM master, no GPU needed -
so they guard the no-bin-specific machinery on every commit. The slow real-data
frozen baseline lives in test_maped_merge_parity.py (needs --runslow + MAPED_TEST_DIR).

Covered:
1. The factorized den (den = sum_i wi (x) wdp, rebuilt by einsum) equals the naive
   full per-tilt accumulation it replaces.
2. The detector-slab cast helper (_grid_sample_tilt) returns the same values for a
   uint16 tilt cast in slabs as a single whole-tilt float grid_sample.
3. End-to-end: the streaming merge is deterministic, and feeding uint16 tilts (the
   detector-slab cast) matches feeding the identical values as float, on a tiny
   synthetic tilt series. There is a single merge path now - the in-memory hold-all
   path was removed, since tilts always arrive sequentially.
"""

import numpy as np
import pytest
import torch

from quantem.diffraction.maped import (
    MAPEDTorch,
    _grid_sample_tilt,
    _shift_diffraction_batch,
    _TiltFiles,
    shift_images_torch,
)


def test_factorized_den_matches_full_accumulation():
    """den rebuilt from per-tilt outer products == naive full 4D accumulation.

    The streaming merge never stores the full (Rout,Cout,Hp,Wp) den; it keeps the
    per-tilt real-space weight wi (Rout,Cout) and diffraction weight wdp (Hp,Wp) and
    rebuilds den = sum_i wi_i (x) wdp_i one band at a time. This must equal summing
    the full outer products directly.
    """
    torch.manual_seed(0)
    n, Rout, Cout, Hp, Wp = 4, 6, 5, 3, 7
    wi = torch.rand(n, Rout, Cout, dtype=torch.float64)
    wdp = torch.rand(n, Hp, Wp, dtype=torch.float64)
    full = torch.zeros(Rout, Cout, Hp, Wp, dtype=torch.float64)
    for i in range(n):
        full += wi[i][:, :, None, None] * wdp[i][None, None, :, :]
    fact = torch.einsum("nrc,nhw->rchw", wi, wdp)
    assert torch.allclose(full, fact, rtol=0, atol=1e-12)


def test_grid_sample_tilt_uint16_slab_matches_float_whole():
    """The uint16 detector-slab path samples the same values as a float whole tilt.

    uint16 -> float is exact, and grid_sample treats detector channels independently,
    so casting in slabs and stitching gives the same interpolated values as one
    whole-tilt float grid_sample (the contiguous-slab kernel can differ only in the
    last fp32 bit, hence atol rather than exact equality).
    """
    torch.manual_seed(0)
    Rs, Cs, H, W, Cout, batch = 8, 8, 4, 5, 6, 7
    a_u16 = (torch.rand(Rs, Cs, H * W) * 2000).to(torch.uint16)
    a_resh_u16 = a_u16.view(Rs, Cs, H * W).permute(2, 0, 1).unsqueeze(0)
    a_resh_f = a_resh_u16.float()
    grid = torch.rand(1, batch, Cout, 2) * 2 - 1
    slab = _grid_sample_tilt(a_resh_u16, grid, torch.float32)  # uint16 -> slab path
    whole = _grid_sample_tilt(a_resh_f, grid, torch.float32)  # float -> whole path
    assert slab.shape == whole.shape == (1, H * W, batch, Cout)
    assert torch.allclose(slab, whole, rtol=1e-6, atol=1e-4)


def test_grid_sample_tilt_float_is_whole_call():
    """Float input takes the single whole-tilt grid_sample (the reference path)."""
    torch.manual_seed(1)
    Rs, Cs, HW, Cout, batch = 6, 6, 9, 4, 5
    a = torch.rand(1, HW, Rs, Cs)
    grid = torch.rand(1, batch, Cout, 2) * 2 - 1
    ref = torch.nn.functional.grid_sample(
        a, grid, mode="bilinear", padding_mode="zeros", align_corners=True
    )
    assert torch.equal(_grid_sample_tilt(a, grid, torch.float32), ref)


def _synthetic_tilts(n=3, Rs=10, Cs=10, H=12, W=12, dtype=torch.float32):
    """Tiny tilt series with a real-space blob + a diffraction disk, each rigidly
    offset per tilt, so preprocess/align have a real (small) signal to lock onto."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:H, 0:W]
    sy, sx = np.mgrid[0:Rs, 0:Cs]
    tilts = []
    for i in range(n):
        # diffraction disk shifted a fraction of a pixel per tilt
        dr, dc = 0.4 * i, -0.3 * i
        disk = np.exp(-(((yy - (H / 2 + dr)) ** 2 + (xx - (W / 2 + dc)) ** 2) / 6.0))
        # real-space brightfield blob shifted per tilt
        br, bc = 0.5 * i, 0.5 * i
        blob = np.exp(-(((sy - (Rs / 2 + br)) ** 2 + (sx - (Cs / 2 + bc)) ** 2) / 8.0))
        vol = (blob[:, :, None, None] * disk[None, None, :, :]) * 1000.0
        vol = vol + 5.0 * rng.standard_normal(vol.shape)
        vol = np.clip(vol, 0, None)
        t = torch.from_numpy(vol).to(dtype if dtype.is_floating_point else torch.float32)
        tilts.append(t.to(dtype) if not dtype.is_floating_point else t.float())
    return tilts


def _align_and_merge(tilts, merge_method="fourier", **merge_kw):
    m = MAPEDTorch.from_datasets(tilts)
    m.preprocess(plot_summary=False)
    m.diffraction_origin(plot_origins=False)
    m.diffraction_align(plot_aligned=False)
    m.real_space_align(num_iter=5, padding=2, edge_blend=2, pad_val="median",
                       shift_method="bilinear", plot_aligned=False)
    return m, m.merge_datasets(real_space_edge_blend=2, shift_method=merge_method,
                               plot_result=False, **merge_kw).tensor


def test_stream_merge_deterministic():
    """The streaming merge is bit-for-bit reproducible on the same aligned state.

    There is one merge path: tilts are streamed one at a time into a float32
    accumulator with a factorized den (the in-memory hold-all path was removed -
    data arrives sequentially anyway). Re-running the merge on an unchanged alignment
    must give an identical result; a non-deterministic reduction order or a stale
    buffer would break this.
    """
    torch.manual_seed(0)
    tilts = _synthetic_tilts()
    m, first = _align_and_merge([t.clone() for t in tilts])
    m.datasets = [t.clone() for t in tilts]
    second = m.merge_datasets(real_space_edge_blend=2, shift_method="fourier",
                              plot_result=False).tensor
    assert first.shape == second.shape
    assert torch.equal(first, second), "streaming merge is non-deterministic"


def test_stream_uint16_matches_float_input():
    """Detector-slab uint16 cast == whole float input, with alignment held fixed.

    Align ONCE on float tilts, then merge the SAME aligned state twice - once with
    float32 tilts (whole grid_sample) and once with the identical values as uint16
    (cast in detector slabs). Reusing one set of shifts isolates the merge's input-
    dtype handling: alignment is so sensitive that re-deriving shifts from rounded
    uint16 would itself shift the result (see the preprocess fix), which is a
    property of alignment, not of the merge path under test here.
    """
    torch.manual_seed(0)
    # integer-valued counts so float and uint16 hold the SAME values (exact cast)
    tf = [t.round() for t in _synthetic_tilts(dtype=torch.float32)]
    tu = [t.clamp(0, 65535).to(torch.uint16) for t in tf]
    m, ref = _align_and_merge([t.clone() for t in tf])  # float, whole grid_sample
    m.datasets = tu  # reuse the SAME alignment, feed uint16 -> detector-slab path
    got = m.merge_datasets(real_space_edge_blend=2, shift_method="fourier",
                           plot_result=False).tensor
    r, g = ref.double(), got.double()
    assert torch.allclose(g, r, rtol=1e-5, atol=1e-3)


@pytest.mark.parametrize("method", ["bilinear", "fourier"])
def test_merge_runs_both_shift_methods(method):
    """Both diffraction shift methods produce a finite merge.

    'bilinear' is the DEFAULT; a refactor that extracted the shifter once crashed it
    (it passed ``ramps[i]``, which only exists for 'fourier') - a NameError that the
    fourier-only tests never hit. This guards that BOTH methods stay callable.
    """
    torch.manual_seed(0)
    _, out = _align_and_merge(_synthetic_tilts(), merge_method=method)
    assert torch.isfinite(out).all(), f"{method} merge produced non-finite values"


def test_shift_diffraction_batch_bilinear_matches_per_image():
    """The batched bilinear shift == the canonical per-image shift_images_torch.

    The merge shifts all (batch, Cout) detector images in one batched call instead of
    a per-image Python loop. grid_sample samples each image independently, so the two
    must agree; this also pins that the merge uses the SAME shift convention as the
    rest of the pipeline (shift_images_torch), not a hand-rolled grid.
    """
    torch.manual_seed(0)
    batch, cout, hp, wp = 4, 5, 12, 12
    dp = torch.rand(batch, cout, hp, wp)
    shift = torch.tensor([[0.37, -0.42]])
    got = _shift_diffraction_batch(dp, "bilinear", None, shift, batch, cout, hp, wp)
    ref = torch.stack([
        torch.stack([
            shift_images_torch(dp[b, c].unsqueeze(0), shift, mode="bilinear").squeeze(0)
            for c in range(cout)
        ])
        for b in range(batch)
    ])
    assert torch.allclose(got, ref, atol=1e-6), "batched bilinear shift != per-image"


def test_fourier_shift_rings_bilinear_does_not():
    """A fourier sub-pixel shift of a sharp bright disk rings; bilinear does not.

    The fourier shift is convolution with a sinc kernel: its oscillating tails ring on
    a sharp high-contrast feature (the direct-beam disk), dipping the result BELOW zero
    and leaving the dotted 'cross' through the center of the merged mean pattern. A
    bilinear (triangle-kernel) shift cannot undershoot a non-negative input. This is
    why 'bilinear' is the default; the test guards that the two behave differently.
    """
    h = w = 64
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    r2 = (yy - h / 2) ** 2 + (xx - w / 2) ** 2
    disk = (r2 < 25).float()[None, None] * 1000.0  # sharp bright central disk
    kr = torch.fft.fftfreq(h)[:, None]
    kc = torch.fft.fftfreq(w)[None, :]
    ramp = torch.exp(-2j * torch.pi * (kr * 0.4 + kc * 0.4))
    shift = torch.tensor([[0.4, 0.4]])
    fourier = _shift_diffraction_batch(disk, "fourier", ramp, shift, 1, 1, h, w)[0, 0]
    bilinear = _shift_diffraction_batch(disk, "bilinear", None, shift, 1, 1, h, w)[0, 0]
    far = r2 > 200  # away from the disk: only ringing lives here
    assert float(fourier[far].min()) < -1.0, "fourier shift should ring below zero"
    assert float(bilinear[far].min()) >= -1e-3, "bilinear shift should not ring"


def test_tiltfiles_releases_previous_before_next():
    """_TiltFiles holds ONE tilt at a time: each read is preceded by a release.

    The whole point of from_files is that a 135 GB seven-tilt series never sits on the
    GPU at once - the loader frees the current tilt before reading the next. This pins
    that contract: indexing tilt i releases tilt i-1 first, in that order.
    """
    log = []

    def read(path):
        log.append(("read", path))
        return torch.zeros(2, 2)

    tf = _TiltFiles(["a", "b", "c"], read)
    real_release = tf.release
    tf.release = lambda: (log.append(("release",)), real_release())[1]
    for i in range(3):
        _ = tf[i]
    assert log == [
        ("release",), ("read", "a"),
        ("release",), ("read", "b"),
        ("release",), ("read", "c"),
    ]


def test_single_tilt_merge_runs():
    """A one-tilt merge runs (n=1 edge case): diffraction_align's i=1..n loop is empty,
    so tilt 0 is the reference and the merge is just its aligned, weighted self."""
    torch.manual_seed(0)
    _, out = _align_and_merge(_synthetic_tilts(n=1), merge_method="bilinear")
    assert torch.isfinite(out).all()


def test_merge_reads_each_file_backed_tilt_once():
    """The merge reads each file-backed tilt EXACTLY ONCE - no extra reads for a
    shape/dtype check or a cross-tilt shape validation.

    A no-bin tilt is 19 GB; re-reading one just to inspect ``.shape`` or ``.dtype``,
    or iterating all of them to validate shapes, would dominate the merge. The shapes
    come from the preprocess summaries (im_bf, dp_mean) instead. This pins one read
    per tilt during the merge.
    """
    torch.manual_seed(0)
    tilts = _synthetic_tilts()
    loads = {"n": 0}

    def read(path):
        loads["n"] += 1
        return tilts[int(path)]

    m = MAPEDTorch.from_datasets([torch.zeros(2, 2, 12, 12) for _ in tilts])
    m.datasets = _TiltFiles(list(range(len(tilts))), read)
    m.preprocess(plot_summary=False)
    m.diffraction_origin(plot_origins=False)
    m.diffraction_align(plot_aligned=False)
    m.real_space_align(num_iter=3, padding=2, edge_blend=2, pad_val="median",
                       shift_method="bilinear", plot_aligned=False)
    before = loads["n"]
    m.merge_datasets(shift_method="bilinear", plot_result=False)
    added = loads["n"] - before
    assert added == len(tilts), (
        f"merge read {added} times for {len(tilts)} tilts - should be one read each"
    )
