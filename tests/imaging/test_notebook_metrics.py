"""Frozen baselines for the headline metrics of notebooks/drift/api/*.ipynb.

Each notebook ends with a quantitative metric (MAE reduction, NCC gap
closure, etc.).  A refactor that silently drops one of those metrics
from, say, 92% to 60% won't fail any unit test — but it *will* make the
demos look broken to every downstream user.

These tests re-run each notebook's exact pipeline in isolation and
assert the headline number stays within ±ε of the published baseline.

Synthetic notebooks (03, 04) are always run.  Real-data notebooks
(01 — Cedric `.npy`, 02 — Cedric subset, 05 — CaSiO3 EDS `.emd`) require
files on disk and are gated by file existence + slow marker.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from quantem.imaging import DriftCorrection

# Published baselines captured from the executed notebooks.
# Tolerances are loose enough for GPU non-determinism but tight enough
# to catch a real algorithmic regression.


# ---------------------------------------------------------------------------
# 03: paired 0°/90° 4D-STEM merge — synthetic, always runs
# ---------------------------------------------------------------------------

def _build_notebook_03_cubes(seed=42):
    """Replicate the chevron pattern + drift simulation from notebook 03."""
    from scipy.ndimage import gaussian_filter
    np.random.seed(seed)
    SCAN, BASE, n_Q = 128, 200, 16
    xa, ya = np.meshgrid(
        np.arange(-BASE // 2, BASE // 2),
        np.arange(-BASE // 2, BASE // 2), indexing="ij",
    )
    base = (np.mod(np.abs(xa) + np.abs(ya), 16) < 8).astype("float")
    base[np.logical_and(xa > 0, ya > 0)] += 0.5
    base[np.maximum(np.abs(xa), np.abs(ya)) < 20] = 2
    base = gaussian_filter(base, sigma=0.667).astype(np.float32)
    det_c = n_Q // 2
    qy, qx = np.meshgrid(np.arange(n_Q) - det_c, np.arange(n_Q) - det_c,
                         indexing="ij")
    q_rad = np.sqrt(qy**2 + qx**2)
    vdf_mask = q_rad > (n_Q // 4)
    dp_template = np.exp(-q_rad**2 / (2 * 6**2))
    u = np.arange(SCAN, dtype=np.float32)
    y_drift = u * 0.1
    jitter0 = np.random.randn(2, SCAN).astype(np.float32) * 0.5
    jitter1 = np.random.randn(2, SCAN).astype(np.float32) * 0.5
    x0_state = [0.0]  # carried over between scans (matches drift_original.ipynb)

    def render():
        cube_0 = np.zeros((SCAN, SCAN, n_Q, n_Q), dtype=np.float32)
        cube_90 = np.zeros_like(cube_0)
        for a0 in range(SCAN):
            x0 = 40 + a0 + jitter0[0, a0]
            y0 = 30 + y_drift[a0] + jitter0[1, a0]
            x = np.clip(np.full(SCAN, x0), 0, BASE - 2)
            y = np.clip(y0 + u, 0, BASE - 2)
            xf, yf = np.floor(x).astype(int), np.floor(y).astype(int)
            dx, dy = x - xf, y - yf
            line = (base[xf, yf] * (1 - dx) * (1 - dy)
                    + base[xf + 1, yf] * dx * (1 - dy)
                    + base[xf, yf + 1] * (1 - dx) * dy
                    + base[xf + 1, yf + 1] * dx * dy)
            cube_0[a0] = dp_template[None] * line[:, None, None] * 0.01 + line[:, None, None] * 0.001
            x0_state[0] = x0
        for a0 in range(SCAN):
            y0 = 30 + a0 + y_drift[a0] + jitter1[1, a0]
            x = np.clip(x0_state[0] - u, 0, BASE - 2)
            y = np.clip(np.full(SCAN, y0), 0, BASE - 2)
            xf, yf = np.floor(x).astype(int), np.floor(y).astype(int)
            dx, dy = x - xf, y - yf
            line = (base[xf, yf] * (1 - dx) * (1 - dy)
                    + base[xf + 1, yf] * dx * (1 - dy)
                    + base[xf, yf + 1] * (1 - dx) * dy
                    + base[xf + 1, yf + 1] * dx * dy)
            cube_90[a0] = dp_template[None] * line[:, None, None] * 0.01 + line[:, None, None] * 0.001
        return cube_0, cube_90, vdf_mask

    return render()


def test_notebook_03_from_4dstem_ncc_gap_closure():
    """Notebook 03 pins ``NCC improvement: ~92.4% of remaining gap``."""
    cube_0, cube_90, vdf_mask = _build_notebook_03_cubes()
    dc = DriftCorrection(cube_0, cube_90, scan_direction_degrees=[0, 90])
    dc.preprocess(pad_fraction=0.25, pad_value="median", kde_sigma=0.5,
                  number_knots=1, normalize=True,
                  show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=11, max_image_shift=64,
                    show_merged=False, show_images=False)
    dc.align_nonrigid(show_merged=False, show_images=False)
    result = dc.generate_corrected(verbose=False)
    m = 10
    SCAN = cube_0.shape[0]
    s = slice(m, SCAN - m)
    vdf_raw_a = cube_0[:, :, vdf_mask].sum(-1)
    vdf_raw_b = np.rot90(cube_90[:, :, vdf_mask].sum(-1), 1)
    vdf_cor_a = result.corrected_a[:, :, vdf_mask].sum(-1)
    vdf_cor_b = result.corrected_b[:, :, vdf_mask].sum(-1)
    ncc_raw = float(np.corrcoef(vdf_raw_a[s, s].ravel(), vdf_raw_b[s, s].ravel())[0, 1])
    ncc_cor = float(np.corrcoef(vdf_cor_a[s, s].ravel(), vdf_cor_b[s, s].ravel())[0, 1])
    gap_closure = (ncc_cor - ncc_raw) / (1 - ncc_raw) * 100
    assert gap_closure >= 85.0, (
        f"Notebook 03 NCC gap closure regressed: {gap_closure:.1f}% < 85% baseline")


# ---------------------------------------------------------------------------
# 04: HAADF reference + drifted 4D-STEM cube — synthetic, always runs
# ---------------------------------------------------------------------------

def _build_notebook_04_data(seed=0):
    from scipy.ndimage import gaussian_filter
    np.random.seed(seed)
    SCAN, BASE, n_Q = 128, 200, 16
    xa, ya = np.meshgrid(np.arange(-BASE // 2, BASE // 2),
                         np.arange(-BASE // 2, BASE // 2), indexing="ij")
    base = (np.mod(np.abs(xa) + np.abs(ya), 16) < 8).astype("float")
    base[np.logical_and(xa > 0, ya > 0)] += 0.5
    base[np.maximum(np.abs(xa), np.abs(ya)) < 20] = 2
    base = gaussian_filter(base, sigma=0.667).astype(np.float32)
    offset = (BASE - SCAN) // 2
    haadf_ref = base[offset:offset + SCAN, offset:offset + SCAN].astype(np.float32)
    u = np.arange(SCAN, dtype=np.float32)
    row_drift = u * 0.001
    col_drift = u * 0.1
    jitter = np.random.randn(2, SCAN).astype(np.float32) * 0.5
    det_c = n_Q // 2
    qy, qx = np.meshgrid(np.arange(n_Q) - det_c, np.arange(n_Q) - det_c,
                         indexing="ij")
    q_rad = np.sqrt(qy**2 + qx**2)
    vdf_mask = q_rad > (n_Q // 4)
    dp_template = np.exp(-q_rad**2 / (2 * 6**2))
    cube_drifted = np.zeros((SCAN, SCAN, n_Q, n_Q), dtype=np.float32)
    for a0 in range(SCAN):
        x0 = 40 + a0 + row_drift[a0] + jitter[0, a0]
        y0 = 30 + col_drift[a0] + jitter[1, a0]
        x = np.clip(np.full(SCAN, x0), 0, BASE - 2)
        y = np.clip(y0 + u, 0, BASE - 2)
        xf, yf = np.floor(x).astype(int), np.floor(y).astype(int)
        dx, dy = x - xf, y - yf
        line = (base[xf, yf] * (1 - dx) * (1 - dy)
                + base[xf + 1, yf] * dx * (1 - dy)
                + base[xf, yf + 1] * (1 - dx) * dy
                + base[xf + 1, yf + 1] * dx * dy)
        cube_drifted[a0] = dp_template[None] * line[:, None, None] * 0.01 + line[:, None, None] * 0.001
    return haadf_ref, cube_drifted, vdf_mask


def test_notebook_04_from_reference_ncc_gap_closure():
    """Notebook 04 pins ``improvement: ~74.3% of remaining gap``."""
    haadf_ref, cube_drifted, vdf_mask = _build_notebook_04_data()
    dc = DriftCorrection(haadf_ref, cube_drifted, scan_direction_degrees=0)
    dc.preprocess(pad_fraction=0.25, pad_value="median", kde_sigma=0.5,
                  number_knots=1, normalize=True,
                  show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=11, max_image_shift=64,
                    show_merged=False, show_images=False)
    dc.align_nonrigid(show_merged=False, show_images=False)
    result = dc.generate_corrected(verbose=False)
    SCAN = haadf_ref.shape[0]
    m = 10
    s = slice(m, SCAN - m)
    vdf_drifted = cube_drifted[:, :, vdf_mask].sum(-1)
    vdf_corrected = result.array[:, :, vdf_mask].sum(-1)
    def znorm(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    ncc_before = float(np.corrcoef(znorm(haadf_ref[s, s]).ravel(),
                                    znorm(vdf_drifted[s, s]).ravel())[0, 1])
    ncc_after = float(np.corrcoef(znorm(haadf_ref[s, s]).ravel(),
                                   znorm(vdf_corrected[s, s]).ravel())[0, 1])
    gap_closure = (ncc_after - ncc_before) / (1 - ncc_before) * 100
    assert gap_closure >= 65.0, (
        f"Notebook 04 NCC gap closure regressed: {gap_closure:.1f}% < 65% baseline")


# ---------------------------------------------------------------------------
# 01 / 02 / 05: real-data, disk-dependent; slow marker
# ---------------------------------------------------------------------------

CEDRIC_DIR = Path("/home/owner/data/cedric/drift/20260207_samsung_GAAFET/dataset_1")
CASIO3_DIR = Path("/home/owner/ssd/data/bob/20260324_drift_colin_caitlyn_eds_4dstem")


@pytest.mark.slow
@pytest.mark.skipif(not (CEDRIC_DIR / "0_deg_images.npy").exists(),
                    reason="Cedric Samsung GAAFET data not on disk")
def test_notebook_01_from_pair_mae_reduction():
    """Notebook 01 pins ``reduction: ~32.2%`` MAE."""
    im0 = np.load(CEDRIC_DIR / "0_deg_images.npy")[0]
    im90 = np.load(CEDRIC_DIR / "90_deg_images.npy")[0]
    dc = DriftCorrection(im0, im90, scan_direction_degrees=[0, -90])
    dc.preprocess(pad_fraction=0.25, pad_value="median", kde_sigma=0.5,
                  number_knots=1, show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=11,
                    show_merged=False, show_images=False)
    dc.align_nonrigid(show_merged=False, show_images=False)
    result = dc.generate_corrected(upsample_factor=1, kde_sigma=0.5,
                                    strip_padding=True, show_merged=False)
    corrected = result.array
    im90_rot = np.rot90(im90, k=-1)
    raw_mae = float(np.abs(im0.astype(float) - im90_rot.astype(float)).mean())
    cor_mae = float(np.abs(im0.astype(float) - corrected).mean())
    reduction = (1 - cor_mae / raw_mae) * 100
    assert reduction >= 25.0, (
        f"Notebook 01 MAE reduction regressed: {reduction:.1f}% < 25% baseline")


@pytest.mark.slow
@pytest.mark.skipif(
    not (CASIO3_DIR / ".cache" / "casio3_haadf_ref_1024x1024.npy").exists(),
    reason="CaSiO3 EDS cache not on disk",
)
def test_notebook_05_from_reference_eds_ncc_gap_closure():
    """Notebook 05 pins ``improvement: ~98.6%`` NCC gap closure (HAADF-vs-HAADF)."""
    from rsciio.emd import file_reader
    cache_dir = CASIO3_DIR / ".cache"
    haadf_ref = np.load(cache_dir / "casio3_haadf_ref_1024x1024.npy").astype(np.float32)
    datasets = file_reader(str(CASIO3_DIR /
        "0041-CaSIO3_134hr_exsitu_SI_1.85_Mx_53.9_nm_EDS_HAADF_Diffraction_Nano.emd"))
    haadf_drifted = eds_cube = None
    for ds in datasets:
        d = ds["data"]
        title = ds.get("metadata", {}).get("General", {}).get("title", "")
        if d.ndim == 2 and "HAADF" in title and haadf_drifted is None:
            haadf_drifted = d.astype(np.float32)
        elif d.ndim == 3 and eds_cube is None:
            eds_cube = np.asarray(d)
    dc = DriftCorrection(haadf_ref, eds_cube, scan_direction_degrees=0,
                          alignment_image=haadf_drifted)
    dc.preprocess(pad_fraction=0.25, pad_value="median", kde_sigma=0.5,
                  number_knots=1, normalize=True,
                  show_merged=False, show_images=False)
    dc.align_affine(step=0.02, num_tests=11, max_image_shift=64,
                    show_merged=False, show_images=False)
    dc.align_nonrigid(show_merged=False, show_images=False)
    haadf_corrected = dc.apply_correction(haadf_drifted, image_index=1)
    if hasattr(haadf_corrected, "cpu"):
        haadf_corrected = haadf_corrected.cpu().numpy()
    H, W = haadf_ref.shape
    m = int(0.05 * H)
    s = (slice(m, H - m), slice(m, W - m))
    def znorm(x):
        x = x.astype(np.float64)
        return (x - x.mean()) / (x.std() + 1e-8)
    ncc_before = float(np.corrcoef(znorm(haadf_ref[s]).ravel(),
                                    znorm(haadf_drifted[s]).ravel())[0, 1])
    ncc_after = float(np.corrcoef(znorm(haadf_ref[s]).ravel(),
                                   znorm(haadf_corrected[s]).ravel())[0, 1])
    gap_closure = (ncc_after - ncc_before) / (1 - ncc_before) * 100
    assert gap_closure >= 90.0, (
        f"Notebook 05 NCC gap closure regressed: {gap_closure:.1f}% < 90% baseline")
