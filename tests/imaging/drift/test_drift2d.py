"""2D drift correction reproduces the paper result, on real data.

Frozen automatic-affine baselines for the four 2D tutorials (srtio3, silicon,
co3o4, ws2), the WS2 publication endpoint parity check, and the downsample
screening-path parity check. The raw EMD files remain external to the
QuantEM repository; these tests skip cleanly when the optional real-data
directory is unavailable. Run the contract on GPU 1 with::

    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src pytest -q \
        tests/imaging/drift/test_drift2d.py

Wall-clock time is deliberately not frozen because CUDA initialization and GPU
contention are environmental. The chosen model, knots, masked NCC, and output
statistics are the scientific contract.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch
from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.io import load
from quantem.imaging.drift import DriftCorrection

zarr = pytest.importorskip("zarr")
pytestmark = pytest.mark.drift_realdata


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parity_device() -> str:
    """Use an explicit parity device or the first visible CUDA GPU."""
    if device := os.environ.get("QUANTEM_DRIFT_PARITY_DEVICE"):
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    pytest.skip(
        "real-data drift parity requires a GPU by default; set "
        "QUANTEM_DRIFT_PARITY_DEVICE=mps or cpu to run on another backend"
    )


def _require(root: Path, *relative_paths: str) -> list[Path]:
    paths = [root / relative_path for relative_path in relative_paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        pytest.fail(
            "drift real-data root exists but required files are missing: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _as_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _image_parity(actual, expected) -> tuple[float, float]:
    actual = _as_numpy(actual).astype(np.float64, copy=False)
    expected = _as_numpy(expected).astype(np.float64, copy=False)
    actual_centered = actual - actual.mean()
    expected_centered = expected - expected.mean()
    denominator = np.linalg.norm(actual_centered) * np.linalg.norm(expected_centered)
    ncc = float(np.vdot(actual_centered, expected_centered) / denominator)
    nrmse = float(
        np.linalg.norm(actual - expected) / np.linalg.norm(expected_centered)
    )
    return ncc, nrmse


def _assert_positions_match(
    actual: DriftCorrection,
    expected: DriftCorrection,
    *,
    atol: float,
    max_coverage_mismatch_fraction: float,
) -> None:
    assert len(actual.imgs) == len(expected.imgs)
    for image_index in range(len(actual.imgs)):
        np.testing.assert_allclose(
            actual.probe_positions(image_index, plot=False),
            expected.probe_positions(image_index, plot=False),
            rtol=0.0,
            atol=atol,
        )
    coverage_mismatch = np.mean(actual.coverage_mask() != expected.coverage_mask())
    assert coverage_mismatch <= max_coverage_mismatch_fraction


@dataclass(frozen=True)
class AutomaticAffineBaseline:
    """One raw-data workflow and its frozen automatic affine result."""

    name: str
    paths: tuple[str, str]
    drift_rate: tuple[float, float]
    candidate_evaluations: int
    refine_downsample: int
    final_error: tuple[float, float, float]
    knot_sum: tuple[float, float]
    knot_std: tuple[float, float]
    before_ncc: tuple[float, float, float, float, float]
    affine_ncc: tuple[float, float, float, float, float]
    corrected_mean: float
    corrected_std: float


BASELINES = (
    AutomaticAffineBaseline(
        name="srtio3",
        paths=(
            "srtio3_xeds/0047_20260709_1134_STEM_HAADF_15.0_Mx_6.74_nm_Diffraction.emd",
            "srtio3_xeds/0046_20260709_1134_STEM_HAADF_15.0_Mx_6.74_nm_Diffraction.emd",
        ),
        drift_rate=(-0.005859375000000001, 0.012304687499999998),
        candidate_evaluations=105,
        refine_downsample=4,
        final_error=(296.26129150390625, 296.2613220214844, 296.26129150390625),
        knot_sum=(3160068.2025756836, 3129339.8808288574),
        knot_std=(648.9393152618258, 647.2012097680804),
        before_ncc=(
            0.04753032699227333,
            0.1124471127986908,
            0.04933137446641922,
            -0.028733249753713608,
            0.9647116661071777,
        ),
        affine_ncc=(
            0.8621276617050171,
            0.8645110130310059,
            0.8603744506835938,
            0.8604859709739685,
            0.9647116661071777,
        ),
        corrected_mean=9137.1474609375,
        corrected_std=1709.5238037109375,
    ),
    AutomaticAffineBaseline(
        name="silicon",
        paths=(
            "0041_20260515_1130_15.0_Mx_6.74_nm_Nano_HAADF.emd",
            "0042_20260515_1130_15.0_Mx_6.74_nm_Nano_HAADF.emd",
        ),
        drift_rate=(-0.07246093749999999, 0.13964843750000003),
        candidate_evaluations=225,
        refine_downsample=4,
        final_error=(153.5565643310547, 153.5565643310547, 153.5565643310547),
        knot_sum=(3195499.6249084473, 3093908.4561309814),
        knot_std=(655.3387367427298, 637.1696516784966),
        before_ncc=(
            -0.00030272899311967194,
            0.0016573418397456408,
            -0.0003136808518320322,
            -0.0026746769435703754,
            0.7943592071533203,
        ),
        affine_ncc=(
            0.3924849033355713,
            0.4192958474159241,
            0.40179243683815,
            0.3542129695415497,
            0.7943592071533203,
        ),
        corrected_mean=2794.718994140625,
        corrected_std=352.13885498046875,
    ),
    AutomaticAffineBaseline(
        name="co3o4",
        paths=(
            "co3o4/0102_20260305_0729_STEM_HAADF_5.20_Mx.emd",
            "co3o4/0103_20260305_0730_STEM_HAADF_5.20_Mx.emd",
        ),
        drift_rate=(-0.00019531250000000017, 0.012499999999999997),
        candidate_evaluations=105,
        refine_downsample=4,
        final_error=(0.009576142765581608, 0.00957613717764616, 0.009576148353517056),
        knot_sum=(3122936.8811035156, 7358727.103652954),
        knot_std=(665.2776048590597, 659.6439271231309),
        before_ncc=(
            0.9681231379508972,
            0.9445685148239136,
            0.9395405650138855,
            0.9727229475975037,
            0.9937880039215088,
        ),
        affine_ncc=(
            0.9839752912521362,
            0.969165563583374,
            0.9729825854301453,
            0.9848613142967224,
            0.9937880039215088,
        ),
        corrected_mean=0.29948240518569946,
        corrected_std=0.1420447677373886,
    ),
    AutomaticAffineBaseline(
        name="ws2",
        paths=(
            "ws2/0127_VWS2_9h_g_300kv_STEM_HAADF_5.20_Mx_Diffraction.emd",
            "ws2/0128_VWS2_9h_g_300kv_STEM_HAADF_5.20_Mx_Diffraction.emd",
        ),
        drift_rate=(-0.039843750000000004, 0.010156249999999999),
        candidate_evaluations=150,
        refine_downsample=2,
        final_error=(0.016078311949968338, 0.016078311949968338, 0.016078311949968338),
        knot_sum=(3224608.1297302246, 7257055.796813965),
        knot_std=(673.6639367280404, 641.4085415992819),
        before_ncc=(
            0.0661502406001091,
            0.021739346906542778,
            0.06861342489719391,
            0.07388093322515488,
            0.9278569221496582,
        ),
        affine_ncc=(
            0.24843864142894745,
            0.21993155777454376,
            0.27620789408683777,
            0.21444958448410034,
            0.9278569221496582,
        ),
        corrected_mean=0.08301956206560135,
        corrected_std=0.034084826707839966,
    ),
)


def _tutorial_data_root() -> Path:
    """Locate repository or explicitly configured raw tutorial data."""
    candidates = [Path(__file__).resolve().parents[3] / "data" / "drift"]
    if root := os.environ.get("QUANTEM_DRIFT_TEST_DATA"):
        candidates.insert(0, Path(root).expanduser())
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip("drift tutorial data not found; set QUANTEM_DRIFT_TEST_DATA")


@pytest.mark.slow
@pytest.mark.parametrize("baseline", BASELINES, ids=lambda baseline: baseline.name)
def test_automatic_affine_matches_frozen_tutorial_result(
    baseline: AutomaticAffineBaseline,
):
    """No-parameter affine alignment must preserve every tutorial result."""
    root = _tutorial_data_root()
    paths = tuple(
        root / path
        if "/" in path
        else next(root.rglob(path), root / path)
        for path in baseline.paths
    )
    missing = [path for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"raw tutorial file missing: {missing[0]}")

    correction = DriftCorrection.from_emd(
        *paths,
        verbose=False,
        device=_parity_device(),
    )
    correction.correct_affine(show_combined=False, verbose=False)

    info = correction.affine_search_info
    assert info["downsample_factor"] == 8
    assert info["refine_downsample_factor"] == baseline.refine_downsample
    assert info["candidate_evaluations"] == baseline.candidate_evaluations
    assert info["native_candidate_evaluations"] == 5
    assert info["fallback_reason"] is None
    assert tuple(correction.shape[1:]) == (2560, 2560)
    assert correction.preprocess_info["padding_fraction"] == pytest.approx(0.25)
    np.testing.assert_allclose(
        info["drift_rate_row_col"],
        baseline.drift_rate,
        rtol=0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        np.asarray(correction.error_track[-1])[1:],
        baseline.final_error,
        rtol=2e-6,
        atol=2e-6,
    )
    knots = [
        knot.detach().cpu().numpy().astype(np.float64)
        for knot in correction.knots
    ]
    np.testing.assert_allclose(
        [knot.sum() for knot in knots],
        baseline.knot_sum,
        rtol=2e-7,
        atol=0.2,
    )
    np.testing.assert_allclose(
        [knot.std() for knot in knots],
        baseline.knot_std,
        rtol=2e-6,
        atol=2e-5,
    )

    report = correction.report()
    columns = (
        "Common NCC",
        "Top third",
        "Middle third",
        "Bottom third",
        "Coverage",
    )
    np.testing.assert_allclose(
        report.loc["before", list(columns)],
        baseline.before_ncc,
        rtol=0.0,
        atol=5e-5,
    )
    np.testing.assert_allclose(
        report.loc["affine", list(columns)],
        baseline.affine_ncc,
        rtol=0.0,
        atol=5e-5,
    )

    corrected = np.asarray(
        correction.corrected(
            upsample_factor=1,
        ).array,
        dtype=np.float32,
    )
    assert corrected.shape == (2560, 2560)
    assert float(corrected.mean()) == pytest.approx(
        baseline.corrected_mean,
        rel=2e-6,
    )
    assert float(corrected.std()) == pytest.approx(
        baseline.corrected_std,
        rel=2e-6,
    )


def test_ws2_publication_endpoint_parity(drift_realdata_root: Path):
    """Raw WS2 affine/non-rigid workflow preserves final public endpoints."""
    raw_0, raw_90, expected_path = _require(
        drift_realdata_root,
        "ws2/0127_VWS2_9h_g_300kv_STEM_HAADF_5.20_Mx_Diffraction.emd",
        "ws2/0128_VWS2_9h_g_300kv_STEM_HAADF_5.20_Mx_Diffraction.emd",
        "ws2/drift.zip",
    )
    expected = load(expected_path)
    actual = DriftCorrection.from_emd(
        raw_0,
        raw_90,
        verbose=False,
        device=_device(),
    )
    actual.correct_affine(show_combined=False, show_scans=False, verbose=False)
    actual.correct_nonrigid(
        optimizer="adam",
        num_refine_cycles=128,
        optimizer_steps=30,
        learning_rate=0.1,
        knot_smoothing_sigma=8.0,
        update_fraction=0.8,
        trend_order=1,
        max_image_shift=32,
        loss="mse",
        early_stop_patience=128,
        min_iterations=4,
        show_combined=False,
        show_scans=False,
        show_knots=False,
        verbose=False,
    )

    _assert_positions_match(
        actual,
        expected,
        atol=0.2,
        max_coverage_mismatch_fraction=1e-4,
    )
    actual_scans = actual.corrected(
        merge=False,
        verbose=False,
    )
    expected_scans = expected.corrected(
        merge=False,
        verbose=False,
    )
    for actual_scan, expected_scan in zip(actual_scans, expected_scans, strict=True):
        assert actual_scan.shape == expected_scan.shape
        ncc, nrmse = _image_parity(actual_scan.array, expected_scan.array)
        assert ncc >= 0.999
        assert nrmse <= 0.03


# ---------------------------------------------------------------------------
# Downsample screening path: a downsample=2 solve must recover the same
# dominant drift as a full-resolution solve, on the same real 2D paper data.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftParityCase:
    """Real-data case and expected full-vs-downsample tolerances."""

    name: str
    cache_name: str
    max_shift_px: int
    affine_step: float
    affine_num_tests: int
    padding_fraction: float
    normalize: bool
    max_total_vector_delta_px: float
    max_total_magnitude_delta_px: float
    max_total_magnitude_delta_nm: float
    min_corrected_ncc: float


@dataclass
class SolvedParity:
    """Compact output from one affine drift solve."""

    drift_rate_px_per_line: np.ndarray
    scan_shape: np.ndarray
    sampling_nm: np.ndarray
    total_original_px: np.ndarray
    total_nm: np.ndarray
    corrected_canvas: np.ndarray
    downsample_metadata: dict
    corrected_metadata: dict

    @property
    def total_original_px_magnitude(self) -> float:
        return float(np.linalg.norm(self.total_original_px))

    @property
    def total_nm_magnitude(self) -> float:
        return float(np.linalg.norm(self.total_nm))


PAPER_2D_CASES = [
    DriftParityCase(
        name="fig2_silicon_before",
        cache_name="fig2_silicon_before.zip",
        max_shift_px=512,
        affine_step=0.02,
        affine_num_tests=21,
        padding_fraction=0.5,
        normalize=False,
        max_total_vector_delta_px=6.0,
        max_total_magnitude_delta_px=5.0,
        max_total_magnitude_delta_nm=0.02,
        min_corrected_ncc=0.88,
    ),
    DriftParityCase(
        name="ws2",
        cache_name="ws2.zip",
        max_shift_px=64,
        affine_step=0.01,
        affine_num_tests=21,
        padding_fraction=0.25,
        normalize=True,
        max_total_vector_delta_px=2.0,
        max_total_magnitude_delta_px=2.0,
        max_total_magnitude_delta_nm=0.02,
        min_corrected_ncc=0.83,
    ),
    DriftParityCase(
        name="coo",
        cache_name="coo.zip",
        max_shift_px=64,
        affine_step=0.01,
        affine_num_tests=21,
        padding_fraction=0.25,
        normalize=True,
        max_total_vector_delta_px=3.0,
        max_total_magnitude_delta_px=1.0,
        max_total_magnitude_delta_nm=0.005,
        min_corrected_ncc=0.98,
    ),
]


def _realdata_cache_dir() -> Path:
    candidates = [Path(__file__).resolve().parents[3] / "data" / "drift" / ".drift_cache"]
    if env := os.environ.get("QUANTEM_DRIFT_REALDATA_DIR"):
        candidates.insert(0, Path(env).expanduser())

    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    pytest.skip(
        "external drift cache is not available; set QUANTEM_DRIFT_REALDATA_DIR "
        "to the .drift_cache folder to run real-data parity tests"
    )


def _load_cache_pair(cache_dir: Path, case: DriftParityCase) -> tuple[list[Dataset2d], list[float]]:
    path = cache_dir / case.cache_name
    if not path.exists():
        pytest.skip(f"external cache file is missing: {path}")

    store = zarr.storage.ZipStore(str(path), mode="r")
    try:
        root = zarr.open_group(store, mode="r")
        datasets = []
        for image_index in range(2):
            array = np.asarray(root[f"imgs/{image_index}/_array"][:], dtype=np.float32)
            sampling = np.asarray(root[f"imgs/{image_index}/_sampling"][:], dtype=float)
            origin = np.asarray(root[f"imgs/{image_index}/_origin"][:], dtype=float)
            datasets.append(
                Dataset2d.from_array(
                    array,
                    name=f"{case.name} scan {image_index}",
                    origin=origin,
                    sampling=sampling,
                    units=["nm", "nm"],
                )
            )
        scan_direction_degrees = [
            float(value) for value in np.asarray(root["scan_direction_degrees"][:]).ravel()
        ]
        return datasets, scan_direction_degrees
    finally:
        close = getattr(store, "close", None)
        if close is not None:
            close()


def _average_downsample_2d(array: np.ndarray, factor: int) -> np.ndarray:
    rows, cols = array.shape
    if rows % factor or cols % factor:
        raise AssertionError(f"cannot average-downsample shape {array.shape} by {factor}")
    return array.reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3))


def _normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_float = np.asarray(a, dtype=np.float64)
    b_float = np.asarray(b, dtype=np.float64)
    a_float -= a_float.mean()
    b_float -= b_float.mean()
    denominator = np.linalg.norm(a_float) * np.linalg.norm(b_float)
    if denominator == 0:
        return 0.0
    return float(np.sum(a_float * b_float) / denominator)


def _solve_affine(
    datasets: list[Dataset2d],
    scan_direction_degrees: list[float],
    case: DriftParityCase,
    *,
    downsample: int,
    device: str,
) -> SolvedParity:
    original_shape = np.asarray(datasets[0].shape[:2], dtype=float)
    original_sampling = np.asarray(datasets[0].sampling[:2], dtype=float)

    drift = DriftCorrection.from_images(
        *datasets,
        scan_direction_degrees=scan_direction_degrees,
        device=device,
    ).preprocess(
        downsample=downsample,
        num_knots=1,
        padding_fraction=case.padding_fraction,
        smoothing_sigma=0.5,
        normalize=case.normalize,
        show_combined=False,
        show_scans=False,
        verbose=False,
    )
    drift.correct_affine(
        max_image_shift=case.max_shift_px,
        max_drift_rate=(
            case.affine_step * ((case.affine_num_tests - 1) / 2)
        ),
        num_rates={21: 21}[case.affine_num_tests],
        refine=True,
        chunk_size=None,
        show_combined=False,
        show_scans=False,
        verbose=False,
    )

    rate = np.asarray(drift.drift_rate, dtype=float)
    shape = np.asarray(drift.imgs[0].shape[:2], dtype=float)
    sampling = np.asarray(drift.imgs[0].sampling[:2], dtype=float)
    total_nm = rate * shape[0] * sampling
    total_original_px = total_nm / original_sampling

    corrected_canvas = drift.corrected(
        output_original_shape=False,
        upsample_factor=1,
    )
    corrected_stripped = drift.corrected(
        output_original_shape=True,
        strip_padding=True,
        upsample_factor=1,
    )

    assert shape[0] * sampling[0] == pytest.approx(original_shape[0] * original_sampling[0])
    assert shape[1] * sampling[1] == pytest.approx(original_shape[1] * original_sampling[1])

    return SolvedParity(
        drift_rate_px_per_line=rate,
        scan_shape=shape,
        sampling_nm=sampling,
        total_original_px=total_original_px,
        total_nm=total_nm,
        corrected_canvas=np.asarray(corrected_canvas.array, dtype=np.float32),
        downsample_metadata=getattr(drift, "downsample_metadata", {}),
        corrected_metadata=dict(corrected_stripped.metadata),
    )


@pytest.mark.slow
@pytest.mark.parametrize("case", PAPER_2D_CASES, ids=lambda case: case.name)
def test_real_paper_2d_downsample_matches_full_resolution_affine(case: DriftParityCase):
    """Downsampled affine solves should match full-resolution solves on paper data.

    The purpose is not to bless downsampled output as publication quality; it is
    to prove that the screening path recovers the same dominant scan drift, keeps
    scale metadata truthful, and produces visually comparable corrected images on
    the real silicon, WS2, and Co3O4 examples used by the drift workflow.
    """
    cache_dir = _realdata_cache_dir()
    device = _parity_device()
    datasets, scan_direction_degrees = _load_cache_pair(cache_dir, case)

    full = _solve_affine(
        datasets,
        scan_direction_degrees,
        case,
        downsample=1,
        device=device,
    )
    downsampled = _solve_affine(
        datasets,
        scan_direction_degrees,
        case,
        downsample=2,
        device=device,
    )

    np.testing.assert_allclose(
        downsampled.scan_shape,
        full.scan_shape / 2,
        err_msg=f"{case.name}: downsample=2 should halve each scan dimension",
    )
    np.testing.assert_allclose(
        downsampled.sampling_nm,
        full.sampling_nm * 2,
        err_msg=f"{case.name}: downsample=2 should double the sampling metadata",
    )

    metadata = downsampled.downsample_metadata
    assert metadata["factor"] == 2
    assert metadata["method"] == "average"
    assert metadata["downsampled_shape"] == [1024, 1024]
    assert metadata["original_images"][0]["shape"] == [2048, 2048]
    assert metadata["original_images"][1]["shape"] == [2048, 2048]

    corrected_metadata = downsampled.corrected_metadata
    assert corrected_metadata["downsample"] == 2
    assert corrected_metadata["downsample_method"] == "average"
    assert corrected_metadata["downsample_metadata"]["original_images"][0]["shape"] == [2048, 2048]
    np.testing.assert_allclose(
        corrected_metadata["downsample_sampling"],
        downsampled.sampling_nm,
    )
    assert corrected_metadata["downsample_units"] == ["nm", "nm"]

    total_vector_delta_px = np.linalg.norm(downsampled.total_original_px - full.total_original_px)
    total_magnitude_delta_px = abs(
        downsampled.total_original_px_magnitude - full.total_original_px_magnitude
    )
    total_magnitude_delta_nm = abs(downsampled.total_nm_magnitude - full.total_nm_magnitude)
    corrected_ncc = _normalized_cross_correlation(
        _average_downsample_2d(full.corrected_canvas, factor=2),
        downsampled.corrected_canvas,
    )

    assert total_vector_delta_px <= case.max_total_vector_delta_px, (
        f"{case.name}: downsample drift vector changed by {total_vector_delta_px:.3f} "
        f"original px; full={full.total_original_px}, downsample={downsampled.total_original_px}"
    )
    assert total_magnitude_delta_px <= case.max_total_magnitude_delta_px, (
        f"{case.name}: drift magnitude changed by {total_magnitude_delta_px:.3f} original px"
    )
    assert total_magnitude_delta_nm <= case.max_total_magnitude_delta_nm, (
        f"{case.name}: drift magnitude changed by {total_magnitude_delta_nm:.4f} nm"
    )
    assert corrected_ncc >= case.min_corrected_ncc, (
        f"{case.name}: corrected-image NCC {corrected_ncc:.4f} is below threshold"
    )
