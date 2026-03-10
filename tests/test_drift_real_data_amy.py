"""Regression tests: PyTorch align_drift vs SciPy align_affine on Amy's 930 kx real data.

Two tiers:
  - Fast (default): bins images to 256×256 before running — completes in seconds.
  - Slow (--runslow): runs on the full 2048×2048 images.

Run fast tests:
    pytest tests/test_drift_real_data_amy.py -v

Run all (including slow):
    pytest tests/test_drift_real_data_amy.py --runslow -v
"""

import numpy as np
import pytest

DATA_DIR = "/home/bobleesj/data/amy"
FILE_0DEG = f"{DATA_DIR}/0008-20260222_gas_light_chip_6_GC40531_500_660_nm_50C_STEM_HAADF_930_kx_Diffraction.emd"
FILE_90DEG = f"{DATA_DIR}/0011-20260222_gas_light_chip_6_GC40531_500_660_nm_50C_STEM_HAADF_930_kx_Diffraction.emd"
SCAN_DIRS = [0, -90]

amy_data_available = pytest.mark.skipif(
    not (
        __import__("pathlib").Path(FILE_0DEG).exists()
        and __import__("pathlib").Path(FILE_90DEG).exists()
    ),
    reason="Amy 930 kx EMD files not found at /home/bobleesj/data/amy/",
)


def _bin_image(img: np.ndarray, factor: int) -> np.ndarray:
    """Block-mean bin a 2D image by an integer factor."""
    h, w = img.shape
    h2, w2 = (h // factor) * factor, (w // factor) * factor
    return img[:h2, :w2].reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3)).astype(np.float32)


def _run_scipy(im0, im1):
    from quantem.core.config import set_device
    from quantem.imaging import DriftCorrection
    set_device("gpu")
    dc = DriftCorrection.from_data(
        images=[im0, im1],
        scan_direction_degrees=SCAN_DIRS,
    ).preprocess(pad_fraction=0.25, number_knots=1)
    dc.align_affine(show_merged=False, show_images=False, show_knots=False)
    merged = dc.generate_corrected_image(show_image=False).array
    n_rows = dc.knots[0].shape[1]
    u = np.arange(n_rows) - (n_rows - 1) / 2
    dx = np.polyfit(u, dc.knots[0][0, :, 0], 1)[0]
    dy = np.polyfit(u, dc.knots[0][1, :, 0], 1)[0]
    return {"merged": merged, "drift_x": dx, "drift_y": dy}


def _run_pytorch(im0, im1):
    import torch
    from quantem.core.config import set_device
    from quantem.imaging.drift_torch import align_drift
    set_device("gpu")
    torch.cuda.empty_cache()
    merged, info = align_drift(
        images=[im0, im1],
        scan_direction_degrees=SCAN_DIRS,
        pad_fraction=0.25,
        verbose=False,
    )
    dx, dy = info["params"]["affine_drift"]
    return {"merged": merged, "drift_x": float(dx), "drift_y": float(dy)}


def _assert_agreement(scipy_r, pytorch_r, drift_tol=0.02, nrmse_tol=0.05, mean_tol=0.02):
    # Shape
    assert scipy_r["merged"].shape == pytorch_r["merged"].shape, (
        f"Shape mismatch: SciPy {scipy_r['merged'].shape} vs PyTorch {pytorch_r['merged'].shape}"
    )
    # Drift x
    dx_diff = abs(scipy_r["drift_x"] - pytorch_r["drift_x"])
    assert dx_diff < drift_tol, (
        f"drift_x: SciPy={scipy_r['drift_x']:.4f}, PyTorch={pytorch_r['drift_x']:.4f}, diff={dx_diff:.4f} >= {drift_tol}"
    )
    # Drift y
    dy_diff = abs(scipy_r["drift_y"] - pytorch_r["drift_y"])
    assert dy_diff < drift_tol, (
        f"drift_y: SciPy={scipy_r['drift_y']:.4f}, PyTorch={pytorch_r['drift_y']:.4f}, diff={dy_diff:.4f} >= {drift_tol}"
    )
    # Normalised RMSE
    s, p = scipy_r["merged"], pytorch_r["merged"]
    rmse = float(np.sqrt(np.mean((s - p) ** 2)))
    sig_std = float(np.std(s))
    nrmse = rmse / max(sig_std, 1e-6)
    assert nrmse < nrmse_tol, (
        f"Normalised RMSE {nrmse:.4f} >= {nrmse_tol} (RMSE={rmse:.2f}, std={sig_std:.2f})"
    )
    # Mean intensity
    mean_s, mean_p = float(np.mean(s)), float(np.mean(p))
    rel = abs(mean_s - mean_p) / max(abs(mean_s), 1e-6)
    assert rel < mean_tol, (
        f"Mean differs by {rel*100:.2f}%: SciPy={mean_s:.1f}, PyTorch={mean_p:.1f}"
    )


# ---------------------------------------------------------------------------
# Fast tier: 256×256 binned images (runs without --runslow)
# ---------------------------------------------------------------------------

@amy_data_available
class TestAmyDriftAgreementFast:
    """PyTorch vs SciPy on 256×256 binned Amy images — fast, no --runslow needed."""

    @pytest.fixture(scope="class")
    def images_256(self):
        from quantem.widget import IO
        im0 = _bin_image(IO.file(FILE_0DEG).data, factor=8)   # 2048 → 256
        im1 = _bin_image(IO.file(FILE_90DEG).data, factor=8)
        assert im0.shape == (256, 256)
        return im0, im1

    def test_output_shapes_match(self, images_256):
        im0, im1 = images_256
        s = _run_scipy(im0, im1)
        p = _run_pytorch(im0, im1)
        assert s["merged"].shape == p["merged"].shape

    def test_drift_x_agreement(self, images_256):
        im0, im1 = images_256
        s = _run_scipy(im0, im1)
        p = _run_pytorch(im0, im1)
        dx_diff = abs(s["drift_x"] - p["drift_x"])
        assert dx_diff < 0.02, (
            f"drift_x: SciPy={s['drift_x']:.4f}, PyTorch={p['drift_x']:.4f}, diff={dx_diff:.4f}"
        )

    def test_drift_y_agreement(self, images_256):
        im0, im1 = images_256
        s = _run_scipy(im0, im1)
        p = _run_pytorch(im0, im1)
        dy_diff = abs(s["drift_y"] - p["drift_y"])
        assert dy_diff < 0.02, (
            f"drift_y: SciPy={s['drift_y']:.4f}, PyTorch={p['drift_y']:.4f}, diff={dy_diff:.4f}"
        )

    def test_merged_image_similarity(self, images_256):
        im0, im1 = images_256
        s = _run_scipy(im0, im1)
        p = _run_pytorch(im0, im1)
        _assert_agreement(s, p)


# ---------------------------------------------------------------------------
# Slow tier: full 2048×2048 images (requires --runslow)
# ---------------------------------------------------------------------------

@amy_data_available
@pytest.mark.slow
class TestAmyDriftAgreementFull:
    """PyTorch vs SciPy on full 2048×2048 Amy images — requires --runslow."""

    @pytest.fixture(scope="class")
    def images_full(self):
        from quantem.widget import IO
        im0 = IO.file(FILE_0DEG).data
        im1 = IO.file(FILE_90DEG).data
        assert im0.shape == (2048, 2048)
        return im0, im1

    @pytest.fixture(scope="class")
    def results_full(self, images_full):
        im0, im1 = images_full
        return _run_scipy(im0, im1), _run_pytorch(im0, im1)

    def test_output_shapes_match(self, results_full):
        _assert_agreement(*results_full)

    def test_drift_x_agreement(self, results_full):
        s, p = results_full
        dx_diff = abs(s["drift_x"] - p["drift_x"])
        assert dx_diff < 0.02, (
            f"drift_x: SciPy={s['drift_x']:.4f}, PyTorch={p['drift_x']:.4f}"
        )

    def test_drift_y_agreement(self, results_full):
        s, p = results_full
        dy_diff = abs(s["drift_y"] - p["drift_y"])
        assert dy_diff < 0.02, (
            f"drift_y: SciPy={s['drift_y']:.4f}, PyTorch={p['drift_y']:.4f}"
        )

    def test_merged_image_similarity(self, results_full):
        _assert_agreement(*results_full)
