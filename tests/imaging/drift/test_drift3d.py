"""3D (XEDS spectrum image) drift correction reproduces the paper result.

Frozen regional NCC floors from the 348-trial dual-GPU sweep on the SrTiO3
XEDS dataset (0048): the same numbers ``reproduce_eds0048.py`` checks. Skips
cleanly without the publication data.
"""


import os
from pathlib import Path

import numpy as np
import pytest
import torch

from quantem.core.io import load
from quantem.imaging import read_emd_eds
from quantem.imaging.drift import DriftCorrection, StripPass
from quantem.imaging.drift.core import strip

pytestmark = pytest.mark.drift_realdata

def _parity_device() -> str:
    if device := os.environ.get("QUANTEM_DRIFT_PARITY_DEVICE"):
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    pytest.skip(
        "GPU required by default; set QUANTEM_DRIFT_PARITY_DEVICE=cpu to force CPU"
    )


def _srtio3_xeds_data_root() -> Path:
    candidates = [
        Path(__file__).resolve().parents[3] / "data" / "drift" / "srtio3_xeds"
    ]
    if env := os.environ.get("QUANTEM_SRTIO3_XEDS_DIR"):
        candidates.insert(0, Path(env).expanduser())
    for path in candidates:
        if path.is_dir() and any(path.glob("0048*.emd")):
            return path
    pytest.skip(
        "SrTiO3 XEDS data not found; set QUANTEM_SRTIO3_XEDS_DIR to its data directory"
    )


def _as_np(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)


def _eds_reference_register(
    data_root: Path,
    eds_glob: str,
    ref0_glob: str,
    ref90_glob: str,
    *,
    device: str,
) -> tuple[DriftCorrection, np.ndarray, np.ndarray, dict]:
    """Shared automatic EDS recipe: solve 0/90 reference, then register SI."""
    eds_path = next(data_root.glob(eds_glob))
    ref0_path = next(data_root.glob(ref0_glob))
    ref90_path = next(data_root.glob(ref90_glob))

    eds = read_emd_eds(eds_path)
    haadf_drifted = eds["haadf"]
    haadf_d = _as_np(
        haadf_drifted.array if hasattr(haadf_drifted, "array") else haadf_drifted
    )
    dc_ref = DriftCorrection.from_emd(
        ref0_path,
        ref90_path,
        device=device,
        verbose=False,
    )
    dc_ref.correct_affine(
        show_combined=False,
        show_scans=False,
        verbose=False,
    )
    dc = DriftCorrection.from_reference(
        dc_ref,
        haadf_drifted,
        scan_direction_degrees=float(eds["scan_rotation_deg"]),
        device=device,
    )
    assert dc._reference_mode
    haadf_ref = np.asarray(dc.imgs[0].array, dtype=np.float32)
    dc.correct_affine(
        show_combined=False,
        show_scans=False,
        verbose=False,
    )

    mask = np.asarray(dc.coverage_mask(), dtype=bool)
    moving_raw = np.asarray(
        getattr(haadf_d, "array", haadf_d), dtype=np.float32
    )
    moving_affine = dc.apply_correction(moving_raw, image_index=1)
    ncc_affine = strip.region_ncc(
        haadf_ref, moving_affine, mask, device=device
    )
    return dc, haadf_ref, haadf_d, ncc_affine

def _strip_multipass_winner(dc: DriftCorrection) -> None:
    """One public call matching the exact 0048 dual-GPU winner recipe."""

    dc.correct_strip(
        passes=[
            StripPass(
                num_strips=24,
                smoothing_sigma=12.0,
                max_column_shift=80,
                max_row_shift=8,
            ),
            StripPass(
                num_strips=24,
                smoothing_sigma=12.0,
                max_column_shift=12,
                max_row_shift=3,
            ),
            StripPass(
                num_strips=64,
                smoothing_sigma=6.0,
                max_column_shift=6,
                max_row_shift=2,
                update_fraction=0.8,
            ),
        ],
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )


@pytest.mark.slow
def test_eds_0048_correct_strip_multipass_ncc_floor():
    """SrTiO3 XEDS 0048: strip correction lifts common NCC vs fixed HAADF ref."""
    data_root = _srtio3_xeds_data_root()
    device = _parity_device()
    if not any(data_root.glob("0048*.emd")):
        pytest.skip("0048 EDS emd missing")

    dc, reference, drifted, ncc_affine = _eds_reference_register(
        data_root,
        "0048*.emd",
        "0047*.emd",
        "0046*.emd",
        device=device,
    )
    assert ncc_affine["common"] >= 0.45, (
        f"0048 affine floor failed: common={ncc_affine['common']:.4f}"
    )

    mask = np.asarray(dc.coverage_mask(), dtype=bool)
    _strip_multipass_winner(dc)
    drifted_array = np.asarray(
        getattr(drifted, "array", drifted), dtype=np.float32
    )
    moving_strip = dc.apply_correction(drifted_array, image_index=1)
    ncc_strip = strip.region_ncc(
        reference, moving_strip, mask, device=device
    )

    # Frozen from the 348-trial dual-GPU sweep. These regional values catch
    # regressions that a permissive whole-image floor can miss, especially a
    # bottom-of-scan collapse hidden by the stronger top lattice.
    expected = {
        "common": 0.8552221,
        "top": 0.9128084,
        "middle": 0.8649281,
        "bottom": 0.8310562,
        # Automatic affine retains slightly more common image area than the
        # historical explicit 121-rate grid (0.753269 versus 0.752271).
        "mask_frac": 0.7532690,
    }

    assert ncc_strip["common"] >= 0.75, (
        f"0048 strip common NCC {ncc_strip['common']:.4f} < 0.75"
    )
    assert ncc_strip["common"] >= ncc_affine["common"] + 0.10, (
        f"0048 strip gain too small: "
        f"{ncc_affine['common']:.4f} -> {ncc_strip['common']:.4f}"
    )
    for region in ("common", "top", "middle", "bottom"):
        assert ncc_strip[region] == pytest.approx(
            expected[region], abs=0.003
        ), (
            f"0048 strip {region} changed: "
            f"{ncc_strip[region]:.6f} != {expected[region]:.6f} +/- 0.003"
        )
    assert ncc_strip["mask_frac"] == pytest.approx(
        expected["mask_frac"], abs=1e-4
    )


@pytest.mark.slow
def test_xeds_publication_endpoint_parity(drift_realdata_root: Path):
    """Preserve the published XEDS positions and corrected element maps."""
    data_root = drift_realdata_root / "srtio3_xeds"
    names = (
        "0047_20260709_1134_STEM_HAADF_15.0_Mx_6.74_nm_Diffraction.emd",
        "0046_20260709_1134_STEM_HAADF_15.0_Mx_6.74_nm_Diffraction.emd",
        "0048_20260709_1134_SI_HAADF_15.0_Mx_6.74_nm_Diffraction.emd",
        "eds_0048_sto_window_maps.npz",
        "eds_drift.zip",
    )
    paths = [data_root / name for name in names]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        pytest.fail(
            "drift real-data root exists but required XEDS files are missing: "
            + ", ".join(str(path) for path in missing)
        )
    ref_0, ref_90, eds_path, maps_path, expected_path = paths

    actual, _, _, _ = _eds_reference_register(
        data_root,
        eds_path.name,
        ref_0.name,
        ref_90.name,
        device=_parity_device(),
    )
    _strip_multipass_winner(actual)
    expected = load(expected_path)

    for image_index in range(len(actual.imgs)):
        np.testing.assert_allclose(
            actual.probe_positions(image_index, plot=False),
            expected.probe_positions(image_index, plot=False),
            rtol=0.0,
            atol=5e-4,
        )
    coverage_mismatch = np.mean(
        actual.coverage_mask() != expected.coverage_mask()
    )
    assert coverage_mismatch <= 1 / expected.coverage_mask().size

    maps = np.load(maps_path)
    for key in ("Ti_K_wide", "Sr_L"):
        image = maps[key].astype(np.float32)
        np.testing.assert_allclose(
            _as_np(actual.apply_correction(image, image_index=1)),
            _as_np(expected.apply_correction(image, image_index=1)),
            rtol=2e-5,
            atol=1e-3,
        )
