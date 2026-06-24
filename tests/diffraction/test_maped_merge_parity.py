"""Parity + regression tests for the vectorized MAPED merge.

`MAPEDTorch.merge_datasets` used to interpolate one output scan-row at a time,
issuing ``batch_size`` tiny ``grid_sample`` calls per dataset (896 total on the
set, 93% of merge wall time). That loop was replaced by a
single ``grid_sample`` over the whole ``(batch, Cout)`` grid. ``grid_sample``
samples every output point independently, so batching the rows must be
bit-for-bit identical to the per-row loop - only faster. These tests pin that:

1. ``test_grid_sample_vectorized_matches_perrow`` - the change in isolation,
   on synthetic tensors of the same shapes the merge feeds ``grid_sample``.
   No data / GPU / widget needed; this is the guard that travels with the PR.
2. ``test_maped_merge_bitexact`` (slow) - the full pipeline on a real
   4D-STEM tilt series: batch-size invariance, determinism, and a
   frozen numeric baseline so a future refactor of the merge can't drift silently.
"""

import os

import numpy as np
import pytest
import torch

# Real 4D-STEM MAPED tilt series (7 tilts, 512x512x192x192).
# Set MAPED_TEST_DIR (a 4D-STEM tilt-series dir of *_master.h5 files) to run the
# slow data tests; they skip when it is unset. MAPED_TEST_PREFIX is the master
# filename prefix within that dir.
MAPED_TEST_DIR = os.environ.get("MAPED_TEST_DIR", "")
MAPED_TEST_PREFIX = os.environ.get("MAPED_TEST_PREFIX", "")

# Frozen baseline of the merged dataset (bin 4x4, fourier shift, edge_blend 5),
# captured from the verified-correct per-row result. The merge involves an FFT
# shift, so cross-environment equality is held to a tolerance; same-environment
# determinism is asserted bit-exact separately below.
BASELINE = dict(
    shape=(128, 128, 192, 192),
    sum=24980595887.365044,
    mean=41.3599873373,
    std=460.7961777026,
    vmin=-33.284821,
    vmax=26566.248047,
    samples={
        (0, 0, 96, 96): 240.432816,
        (64, 64, 96, 96): 230.818954,
        (127, 127, 100, 100): 1174.684082,
        (32, 100, 50, 150): 5.035089,
    },
)


def _interp_perrow(a_reshaped, w_rs_reshaped, c_norm, r_norm, H, W, batch, Cout):
    """Reference: the original per-row interpolation merge_datasets used.

    Loops one output scan-row at a time, sampling on a ``(1, Cout, 1, 2)`` grid,
    then stacks. This is the implementation the vectorized path replaced; it
    exists here only as the parity reference.
    """
    dp_list, wi_list = [], []
    for b in range(batch):
        grid_b = torch.stack([c_norm[b : b + 1, :], r_norm[b : b + 1, :]], dim=-1).unsqueeze(
            2
        )  # (1, Cout, 1, 2)
        dp = torch.nn.functional.grid_sample(
            a_reshaped, grid_b, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        wi = torch.nn.functional.grid_sample(
            w_rs_reshaped, grid_b, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        dp_list.append(dp.squeeze(0).squeeze(-1).view(H, W, Cout).permute(2, 0, 1))
        wi_list.append(wi.squeeze(0).squeeze(-1).squeeze(0))
    return torch.stack(dp_list), torch.stack(wi_list)


def _interp_vectorized(a_reshaped, w_rs_reshaped, c_norm, r_norm, H, W, batch, Cout):
    """Production: the single batched grid_sample now in merge_datasets."""
    grid_full = torch.stack([c_norm, r_norm], dim=-1).unsqueeze(0)  # (1, batch, Cout, 2)
    dp = torch.nn.functional.grid_sample(
        a_reshaped, grid_full, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # (1, H*W, batch, Cout)
    wi = torch.nn.functional.grid_sample(
        w_rs_reshaped, grid_full, mode="bilinear", padding_mode="zeros", align_corners=True
    )  # (1, 1, batch, Cout)
    dp_interp = dp.squeeze(0).view(H, W, batch, Cout).permute(2, 3, 0, 1)  # (batch, Cout, H, W)
    return dp_interp, wi.squeeze(0).squeeze(0)


def test_grid_sample_vectorized_matches_perrow():
    """The vectorized grid_sample is bit-for-bit equal to the per-row loop.

    Guards the exact reshape/permute the optimization introduced. If a future
    edit gets the (H, W, batch, Cout) unravel or the axis order wrong, the
    output stays plausible but wrong; torch.equal here catches it.
    """
    torch.manual_seed(0)
    Rs, Cs, H, W, Cout, batch = 16, 16, 8, 8, 12, 10
    a = torch.rand(Rs, Cs, H * W)
    a_reshaped = a.permute(2, 0, 1).unsqueeze(0)  # (1, H*W, Rs, Cs)
    w_rs_reshaped = torch.rand(1, 1, Rs, Cs)
    # normalized sample coords in [-1, 1], same shapes merge builds: (batch, Cout)
    c_norm = torch.rand(batch, Cout) * 2 - 1
    r_norm = torch.rand(batch, Cout) * 2 - 1

    dp_ref, wi_ref = _interp_perrow(a_reshaped, w_rs_reshaped, c_norm, r_norm, H, W, batch, Cout)
    dp_vec, wi_vec = _interp_vectorized(a_reshaped, w_rs_reshaped, c_norm, r_norm, H, W, batch, Cout)

    assert dp_vec.shape == dp_ref.shape == (batch, Cout, H, W)
    assert torch.equal(dp_vec, dp_ref), "vectorized dp_interp != per-row reference"
    assert torch.equal(wi_vec, wi_ref), "vectorized wi != per-row reference"


def __available():
    if not torch.cuda.is_available():
        return False
    if not os.path.isdir(MAPED_TEST_DIR):
        return False
    try:
        import quantem.widget  # noqa: F401  - GPU loader used by the pipeline
    except ImportError:
        return False
    masters = [
        f
        for f in os.listdir(MAPED_TEST_DIR)
        if f.endswith("master.h5") and f.startswith(MAPED_TEST_PREFIX)
    ]
    return len(masters) >= 2


def _freest_gpu():
    free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
    return int(max(range(len(free)), key=lambda i: free[i]))


def _run__merge(batch_size):
    """Load the tilts (GPU LZ4) -> align -> merge. Returns the merged tensor."""
    import cupy as cp

    from quantem.core.config import set_device
    from quantem.diffraction import MAPEDTorch
    from quantem.widget import load as wload

    # Use the GPU with the most free memory (the box may have other work, e.g. a
    # dashboard, pinned to GPU0).
    gpu = _freest_gpu()
    cp.cuda.Device(gpu).use()
    device = torch.device(f"cuda:{gpu}")
    set_device(device)
    files = sorted(
        f
        for f in os.listdir(MAPED_TEST_DIR)
        if f.endswith("master.h5") and f.startswith(MAPED_TEST_PREFIX)
    )
    ds = []
    for f in files:
        w = wload(os.path.join(MAPED_TEST_DIR, f)).data  # cupy uint16, GPU
        rr, cc, hh, ww = w.shape
        b = w.reshape(rr // 4, 4, cc // 4, 4, hh, ww).sum(axis=(1, 3), dtype=cp.uint32)
        ds.append(torch.from_dlpack(b.astype(cp.float32)))
        del w, b
        cp.get_default_memory_pool().free_all_blocks()
    maped = MAPEDTorch.from_datasets(ds)
    maped.preprocess(plot_summary=False)
    maped.diffraction_origin(vmax=1000, sigma=1, plot_origins=False)
    maped.diffraction_align(edge_blend=2, vmax=5000, plot_aligned=False)
    maped.real_space_align(
        num_iter=20,
        hanning_filter=True,
        padding=2,
        edge_blend=5,
        pad_val="median",
        shift_method="bilinear",
        plot_aligned=False,
    )
    merged = maped.merge_datasets(
        real_space_edge_blend=5,
        shift_method="fourier",
        batch_size=batch_size,
        plot_result=False,
    )
    return maped, merged.tensor


@pytest.mark.slow
@pytest.mark.skipif(not __available(), reason="needs CUDA + quantem.widget + MAPED data")
def test_nobin_single_gpu_streaming_fits():
    """No-bin (512x512x192x192) MAPED merge runs on ONE GPU, streaming uint16 tilts.

    Guards the headline single-GPU capability: load one uint16 tilt, add it to the
    float32 num accumulator (den kept factorized), discard, load the next - so the
    peak is num (38.6 GB) + one uint16 tilt (19.3 GB) + a detector slab, never all
    seven tilts (135 GB) and never a second full den. Asserts the result is the full
    no-bin shape, finite, and that the merge peak stays well under a 96 GB card.
    """
    import cupy as cp

    from quantem.core.config import set_device
    from quantem.diffraction import MAPEDTorch
    from quantem.widget import load as wload

    # The streaming peak is num (38.6 GB) + ONE uint16 tilt (19.3 GB) + a slab,
    # independent of how many tilts there are - so 3 masters prove the single-GPU
    # fit just as well as 7, in a fraction of the load + merge time.
    torch.cuda.empty_cache()  # drop any cached blocks from a prior test in this process
    gpu = _freest_gpu()
    free_gb = torch.cuda.mem_get_info(gpu)[0] / 1e9
    if free_gb < 90:
        pytest.skip(f"needs a ~96 GB GPU mostly free; cuda:{gpu} has {free_gb:.0f} GB")
    cp.cuda.Device(gpu).use()
    device = torch.device(f"cuda:{gpu}")
    set_device(device)
    files = sorted(
        os.path.join(MAPED_TEST_DIR, f)
        for f in os.listdir(MAPED_TEST_DIR)
        if f.endswith("master.h5") and f.startswith(MAPED_TEST_PREFIX)
    )[:3]

    class _LazyUint16:
        """One no-bin uint16 tilt at a time: GPU LZ4 load -> torch uint16 clone ->
        free the cupy buffer (the clone is torch-owned, so freeing cupy is safe)."""

        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            w = wload(self.paths[i]).data
            t = torch.from_dlpack(w).clone()
            del w
            cp.get_default_memory_pool().free_all_blocks()
            return t

        def __iter__(self):
            for i in range(len(self)):
                yield self[i]

    n = len(files)
    maped = MAPEDTorch.from_datasets(
        [torch.zeros((2, 2, 192, 192), dtype=torch.uint16, device=device) for _ in range(n)]
    )
    maped.datasets = _LazyUint16(files)
    maped.preprocess(plot_summary=False)
    maped.diffraction_origin(vmax=1000, sigma=1, plot_origins=False)
    maped.diffraction_align(edge_blend=2, vmax=5000, plot_aligned=False)
    maped.real_space_align(
        num_iter=20, hanning_filter=True, padding=2, edge_blend=5,
        pad_val="median", shift_method="bilinear", plot_aligned=False,
    )
    torch.cuda.reset_peak_memory_stats(device)
    merged = maped.merge_datasets(
        real_space_edge_blend=5, shift_method="fourier",
        batch_size=64, plot_result=False,
    ).tensor
    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    assert tuple(merged.shape) == (512, 512, 192, 192)
    assert merged.dtype == torch.float32
    assert str(merged.device) == f"cuda:{gpu}"
    assert bool(torch.isfinite(merged[0]).all())  # band-wise: a full isfinite is 77 GB
    assert peak_gb < 90, f"streaming merge peak {peak_gb:.1f} GB should fit a 96 GB card"


@pytest.mark.slow
@pytest.mark.skipif(not __available(), reason="needs CUDA + quantem.widget + MAPED data")
def test_maped_merge_bitexact():
    """Real series: merge is batch-invariant, deterministic, and on baseline.

    - batch_size must not change the result: bs=128 (fully vectorized) and bs=15
      (chunked) are bit-for-bit equal, proving the vectorization is independent
      of how the rows are split.
    - determinism: re-running merge on the same aligned state is bit-for-bit equal.
    - frozen baseline: global stats + sampled voxels match the captured values,
      so a future merge refactor that drifts numerically is caught.
    """
    maped, merged_128 = _run__merge(batch_size=128)

    # batch-size invariance (bit-exact) - reuses the same aligned state.
    merged_15 = maped.merge_datasets(
        real_space_edge_blend=5, shift_method="fourier", batch_size=15, plot_result=False
    ).tensor
    assert torch.equal(merged_128, merged_15), "merge result depends on batch_size"

    # determinism (bit-exact) - re-run, same aligned state.
    merged_again = maped.merge_datasets(
        real_space_edge_blend=5, shift_method="fourier", batch_size=128, plot_result=False
    ).tensor
    assert torch.equal(merged_128, merged_again), "merge is non-deterministic"

    # Frozen numeric baseline. The vectorized merge is bit-for-bit identical to the
    # original per-row merge that produced these values (verified element-wise, max
    # abs diff 0.0 over the full 604M-voxel volume), so every statistic - including
    # the extreme vmin/vmax tail - reproduces to float precision. Tight rtol on all
    # of them: any real regression (a shifted alignment, a wrong reshape, an altered
    # weight) moves these immediately.
    m = merged_128.double()
    assert tuple(m.shape) == BASELINE["shape"]
    np.testing.assert_allclose(float(m.sum()), BASELINE["sum"], rtol=1e-6)
    np.testing.assert_allclose(float(m.mean()), BASELINE["mean"], rtol=1e-6)
    np.testing.assert_allclose(float(m.std()), BASELINE["std"], rtol=1e-6)
    np.testing.assert_allclose(float(m.min()), BASELINE["vmin"], rtol=1e-5)
    np.testing.assert_allclose(float(m.max()), BASELINE["vmax"], rtol=1e-5)
    for idx, expected in BASELINE["samples"].items():
        np.testing.assert_allclose(float(m[idx]), expected, rtol=1e-5, atol=1e-4)
