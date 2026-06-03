"""End-to-end parity: widget virtual() / dpc() / center_of_mass() vs quantem.live.

The widget's MPS Metal path and CUDA torch path must produce the SAME numbers as
quantem.live's reference CUDA implementation (engine.dpc / engine.preprocess).
These run on the CUDA box (cupy + quantem.live present); they skip cleanly
elsewhere. Real data (gold_512, pre-binned, ~25 MB from Hugging Face) keeps them
deterministic and fast.
"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytest.importorskip("quantem.live")


@pytest.fixture(scope="module")
def gold():
    from quantem.widget.io import download
    path = download("gold_512_npy_bin4", verbose=False)
    return np.load(f"{path}/data.npy")  # (512, 512, 48, 48) uint16


def test_center_of_mass_matches_quantem_live(gold):
    from quantem.widget.dpc import center_of_mass
    from quantem.live.engine.dpc import compute_center_of_mass
    com_row, com_col = center_of_mass(gold)                      # widget (mean-subtracted)
    ref_row, ref_col = compute_center_of_mass(                   # reference
        cp.asarray(gold.reshape(-1, *gold.shape[-2:])), normalize_zero_mean=True)
    ref_row = cp.asnumpy(ref_row).reshape(512, 512)
    ref_col = cp.asnumpy(ref_col).reshape(512, 512)
    np.testing.assert_allclose(com_row, ref_row, atol=1e-3)
    np.testing.assert_allclose(com_col, ref_col, atol=1e-3)


def test_dpc_phase_matches_quantem_live(gold):
    from quantem.widget.dpc import dpc
    from quantem.live import dpc as live_dpc
    w = dpc(gold, verbose=False)
    r = live_dpc(cp.asarray(gold.reshape(-1, *gold.shape[-2:])), scan_shape=(512, 512), verbose=False)
    assert abs(w.rotation_deg - float(r.rotation_angle_deg)) < 1.0
    ref_phase = cp.asnumpy(r.phase)
    rel = np.abs(w.phase - ref_phase).max() / max(float(np.ptp(ref_phase)), 1e-9)
    assert rel < 1e-3, f"iDPC phase rel diff {rel:.2e}"


def test_virtual_matches_manual_masked_sum(gold):
    """virtual(mode) at an explicit probe == a direct masked-sum over that band."""
    from quantem.widget.detector import virtual, auto_probe, _detector_mask, _resolve_backend
    mean_dp = np.asarray(_resolve_backend(gold).mean_dp(), dtype=np.float32)
    center, r = auto_probe(mean_dp)
    for mode in ("BF", "ABF", "ADF", "HAADF", "DF"):
        vi = virtual(gold, mode, center=center, bf_radius=r)
        mask = _detector_mask(mode, center, r, mean_dp.shape, None, None)
        ref = (gold.astype(np.float64) * mask).sum(axis=(2, 3)).astype(np.float32)
        np.testing.assert_allclose(vi, ref, rtol=1e-4, atol=1.0,
                                   err_msg=f"virtual({mode}) != manual masked-sum")


def test_dataset_container_roundtrip(gold):
    """Dataset4dstemGPU.virtual/.dpc == the standalone functions."""
    from quantem.widget import Dataset4dstemGPU
    from quantem.widget.detector import virtual
    from quantem.widget.dpc import dpc
    ds = Dataset4dstemGPU(gold)
    assert ds.shape == (512, 512, 48, 48)
    # ds.adf() defaults to the r..2r band == standalone virtual("ADF")
    np.testing.assert_array_equal(ds.adf(), virtual(gold, "ADF"))
    # second call is served from cache (same array object), not recomputed
    assert ds.bf() is ds.bf()
    # ds.idpc() == standalone dpc().phase; ds.rotation_deg == standalone rotation
    ref = dpc(gold, verbose=False)
    np.testing.assert_array_equal(ds.idpc(), ref.phase)
    assert abs(ds.rotation_deg - ref.rotation_deg) < 1e-6
    assert ds.idpc() is ds.idpc()                       # iDPC cached
    np.testing.assert_array_equal(ds.com.col, ref.com_col)  # horizontal == com_col
    np.testing.assert_array_equal(ds.com.row, ref.com_row)  # vertical == com_row
