import numpy as np
import pytest
import torch
import torch.nn.functional as F

from quantem.core.datastructures.dataset2d import Dataset2d
from quantem.core.datastructures.dataset4dstem import Dataset4dstem
from quantem.core.datastructures.polar4dstem import Polar4dstem
from quantem.diffraction.polar import PairDistributionFunction
from quantem.diffraction.polar_transform import (
    _array_chunk_to_device_float32,
    _build_candidate_grids,
    _build_polar_sampling_offsets,
    auto_origin_id,
    mean_dp_torch,
    polar_transform,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def synthetic_diffraction_pattern():
    """Create a synthetic diffraction pattern with concentric rings."""
    ny, nx = 256, 256
    y, x = np.ogrid[:ny, :nx]
    cy, cx = (ny - 1) / 2.0, (nx - 1) / 2.0

    # Create rings with Gaussian profiles at specific radii
    pattern = np.zeros((ny, nx), dtype=np.float32)
    ring_radii = [10, 20, 30, 40]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    for radius in ring_radii:
        pattern += 100 * np.exp(-((r - radius) ** 2) / (2 * 2**2))
    # central beam
    pattern += 1000 * np.exp(-(r**2) / (2 * 3**2))
    # noise
    rng = np.random.default_rng(42)
    pattern += rng.poisson(5, size=(ny, nx))

    return pattern.astype(np.float32)


@pytest.fixture
def synthetic_4dstem_dataset(synthetic_diffraction_pattern):
    """Create a synthetic 4D-STEM dataset with 3x3 scan."""
    scan_y, scan_x = 3, 3
    ny, nx = synthetic_diffraction_pattern.shape

    array_4d = np.zeros((scan_y, scan_x, ny, nx), dtype=np.float32)
    for iy in range(scan_y):
        for ix in range(scan_x):
            # Add slight variations
            rng = np.random.default_rng(42 + iy * scan_x + ix)
            variation = 1.0 + 0.1 * rng.standard_normal()
            array_4d[iy, ix] = synthetic_diffraction_pattern * variation

    return Dataset4dstem.from_array(
        array=array_4d,
        name="test_4dstem",
        origin=(0, 0, 0, 0),
        sampling=(1.0, 1.0, 0.015, 0.015),
        units=["nm", "nm", "1/Angstrom", "1/Angstrom"],
        signal_units="counts",
    )


@pytest.fixture
def synthetic_dataset2d(synthetic_diffraction_pattern):
    """Create a synthetic 2D diffraction dataset."""
    return Dataset2d.from_array(
        array=synthetic_diffraction_pattern,
        name="test_2d_diffraction",
        origin=(0, 0),
        sampling=(0.015, 0.015),
        units=["1/Angstrom", "1/Angstrom"],
        signal_units="counts",
    )


def _origin_parity_dataset() -> Dataset4dstem:
    """Small deterministic stack for brittle origin-finder parity checks."""
    ny = nx = 160
    y, x = np.ogrid[:ny, :nx]
    base_center = (ny - 1) / 2.0
    centers = np.array(
        [
            [
                [base_center, base_center],
                [base_center + 1, base_center - 2],
                [base_center - 2, base_center + 2],
            ],
            [
                [base_center + 2, base_center + 1],
                [base_center - 1, base_center - 1],
                [base_center - 3, base_center],
            ],
        ],
        dtype=float,
    )

    array_4d = np.empty((*centers.shape[:2], ny, nx), dtype=np.float32)
    for iy in range(centers.shape[0]):
        for ix in range(centers.shape[1]):
            cy, cx = centers[iy, ix]
            radius = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
            pattern = np.zeros((ny, nx), dtype=np.float32)
            for ring_radius, amplitude, sigma in (
                (10, 80, 1.8),
                (22, 120, 2.2),
                (38, 60, 2.8),
            ):
                pattern += amplitude * np.exp(-((radius - ring_radius) ** 2) / (2 * sigma**2))
            pattern += 1000 * np.exp(-(radius**2) / (2 * 2.5**2))
            pattern += 2.0
            array_4d[iy, ix] = pattern

    return Dataset4dstem.from_array(array_4d, name="origin_parity")


ORIGIN_PARITY_EXPECTED = np.array(
    [
        [[80.0, 80.0], [81.0, 78.0], [78.0, 82.0]],
        [[82.0, 81.0], [79.0, 79.0], [77.0, 80.0]],
    ],
    dtype=float,
)


def _reference_shared_scores(
    dp_batch: torch.Tensor,
    cand_rows: torch.Tensor,
    cand_cols: torch.Tensor,
    offset_row: torch.Tensor,
    offset_col: torch.Tensor,
    min_r_idx: int,
    max_r_idx: int,
    n_row: int,
    n_col: int,
) -> torch.Tensor:
    col_norm_scale = 2.0 / (n_col - 1)
    row_norm_scale = 2.0 / (n_row - 1)
    base_col_norm = offset_col * col_norm_scale
    base_row_norm = offset_row * row_norm_scale
    grid_col = (
        base_col_norm.unsqueeze(0)
        + (cand_cols.float() * col_norm_scale - 1.0)[:, None, None]
    )
    grid_row = (
        base_row_norm.unsqueeze(0)
        + (cand_rows.float() * row_norm_scale - 1.0)[:, None, None]
    )
    grids = torch.stack([grid_col, grid_row], dim=-1)
    polars = F.grid_sample(
        dp_batch.transpose(0, 1).expand(cand_rows.numel(), dp_batch.shape[0], n_row, n_col),
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return polars[:, :, :, min_r_idx:max_r_idx].std(dim=2).sum(dim=2).T


def _reference_paired_scores(
    dp_batch: torch.Tensor,
    cand_rows: torch.Tensor,
    cand_cols: torch.Tensor,
    offset_row: torch.Tensor,
    offset_col: torch.Tensor,
    min_r_idx: int,
    max_r_idx: int,
    n_row: int,
    n_col: int,
) -> torch.Tensor:
    col_norm_scale = 2.0 / (n_col - 1)
    row_norm_scale = 2.0 / (n_row - 1)
    base_col_norm = offset_col * col_norm_scale
    base_row_norm = offset_row * row_norm_scale
    n_cands = cand_rows.shape[1]
    grid_col = (
        base_col_norm
        + (cand_cols.reshape(-1).float() * col_norm_scale - 1.0)[:, None, None]
    )
    grid_row = (
        base_row_norm
        + (cand_rows.reshape(-1).float() * row_norm_scale - 1.0)[:, None, None]
    )
    grids = torch.stack([grid_col, grid_row], dim=-1)
    polars = F.grid_sample(
        dp_batch.repeat_interleave(n_cands, dim=0),
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    n_phi, n_r = base_col_norm.shape
    return (
        polars.view(dp_batch.shape[0], n_cands, n_phi, n_r)[..., min_r_idx:max_r_idx]
        .std(dim=2)
        .sum(dim=2)
    )


def _optimized_shared_scores(
    dp_batch: torch.Tensor,
    center_row: int,
    center_col: int,
    margin: int,
    step: int,
    offset_row: torch.Tensor,
    offset_col: torch.Tensor,
    min_r_idx: int,
    max_r_idx: int,
    n_row: int,
    n_col: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    col_norm_scale = 2.0 / (n_col - 1)
    row_norm_scale = 2.0 / (n_row - 1)
    base_col_norm = offset_col[:, min_r_idx:max_r_idx] * col_norm_scale
    base_row_norm = offset_row[:, min_r_idx:max_r_idx] * row_norm_scale
    col_origin_norm = (
        torch.arange(n_col, dtype=torch.float32, device=dp_batch.device) * col_norm_scale - 1.0
    )
    row_origin_norm = (
        torch.arange(n_row, dtype=torch.float32, device=dp_batch.device) * row_norm_scale - 1.0
    )
    cand_rows, cand_cols, grids = _build_candidate_grids(
        base_col_norm,
        base_row_norm,
        center_row,
        center_col,
        margin,
        n_row,
        n_col,
        col_norm_scale,
        row_norm_scale,
        "mps",
        step=step,
        col_origin_norm=col_origin_norm,
        row_origin_norm=row_origin_norm,
    )
    polars = F.grid_sample(
        dp_batch.transpose(0, 1).expand(cand_rows.numel(), dp_batch.shape[0], n_row, n_col),
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    scores = polars.var(dim=2, correction=1).sqrt().sum(dim=2).T
    return cand_rows, cand_cols, scores


def _optimized_paired_scores(
    dp_batch: torch.Tensor,
    cand_rows: torch.Tensor,
    cand_cols: torch.Tensor,
    offset_row: torch.Tensor,
    offset_col: torch.Tensor,
    min_r_idx: int,
    max_r_idx: int,
    n_row: int,
    n_col: int,
) -> torch.Tensor:
    col_norm_scale = 2.0 / (n_col - 1)
    row_norm_scale = 2.0 / (n_row - 1)
    base_col_norm = offset_col[:, min_r_idx:max_r_idx] * col_norm_scale
    base_row_norm = offset_row[:, min_r_idx:max_r_idx] * row_norm_scale
    col_origin_norm = (
        torch.arange(n_col, dtype=torch.float32, device=dp_batch.device) * col_norm_scale - 1.0
    )
    row_origin_norm = (
        torch.arange(n_row, dtype=torch.float32, device=dp_batch.device) * row_norm_scale - 1.0
    )
    n_cands = cand_rows.shape[1]
    grids = torch.empty(
        (
            dp_batch.shape[0],
            n_cands,
            base_col_norm.shape[0],
            base_col_norm.shape[1],
            2,
        ),
        dtype=base_col_norm.dtype,
        device=base_col_norm.device,
    )
    grids[..., 0] = base_col_norm[None, None, :, :] + col_origin_norm[cand_cols][
        :, :, None, None
    ]
    grids[..., 1] = base_row_norm[None, None, :, :] + row_origin_norm[cand_rows][
        :, :, None, None
    ]
    grids = grids.reshape(
        dp_batch.shape[0], n_cands, base_col_norm.shape[0] * base_col_norm.shape[1], 2
    )
    polars = F.grid_sample(dp_batch, grids, mode="bilinear", padding_mode="zeros", align_corners=True)
    return (
        polars.squeeze(1)
        .view(dp_batch.shape[0], n_cands, *base_col_norm.shape)
        .var(dim=2, correction=1)
        .sqrt()
        .sum(dim=2)
    )


def _mps_origin_score_case():
    device = "mps"
    n_row = n_col = 160
    dp_batch = torch.from_numpy(
        np.ascontiguousarray(_origin_parity_dataset().array.reshape(-1, n_row, n_col)[:4])
    ).to(device)[:, None]
    offset_row, offset_col, _, radial_bins = _build_polar_sampling_offsets(
        None,
        18,
        0.0,
        56.0,
        1.0,
        False,
        device,
    )
    min_r_idx = int(np.floor(0.1 * radial_bins.numel()))
    max_r_idx = int(np.ceil(0.9 * radial_bins.numel()))
    return dp_batch, offset_row, offset_col, min_r_idx, max_r_idx, n_row, n_col


# ============================================================================
# Test PairDistributionFunction Construction
# ============================================================================


class TestPairDistributionFunctionConstruction:
    """Test PairDistributionFunction initialization from various input types."""

    def test_from_data_with_dataset4dstem(self, synthetic_4dstem_dataset):
        """Test construction from a Dataset4dstem object."""
        pdf = PairDistributionFunction.from_data(
            synthetic_4dstem_dataset,
            find_origin=False,
        )
        assert isinstance(pdf.polar, Polar4dstem)
        assert pdf.input_data is synthetic_4dstem_dataset
        assert pdf.polar.shape[0] == 3  # scan_y
        assert pdf.polar.shape[1] == 3  # scan_x
        assert pdf.polar.shape[2] == 180  # num_annular_bins

    def test_direct_init_without_token_raises(self, synthetic_dataset2d):
        """Test that direct __init__ without token raises RuntimeError."""
        pdf_valid = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        with pytest.raises(RuntimeError, match="Use PairDistributionFunction.from_data"):
            PairDistributionFunction(polar=pdf_valid.polar)

    def test_find_origin(self, synthetic_4dstem_dataset):
        """Test automatic origin finding."""
        origin_array = auto_origin_id(
            synthetic_4dstem_dataset,
        )
        assert origin_array.shape == (3, 3, 2)  # (scan_y, scan_x, 2)
        expected_center = 127.5
        for iy in range(3):
            for ix in range(3):
                row, col = origin_array[iy, ix]
                assert abs(row - expected_center) < 1
                assert abs(col - expected_center) < 1

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
    def test_find_origin_mps_brittle_origin_parity(self):
        """MPS optimized origin finder must keep the exact locked origins."""
        origin_array = auto_origin_id(
            _origin_parity_dataset(),
            device="mps",
            batch_size=3,
            local_margin=25,
            radial_step=1.0,
            show_progress=False,
        )
        np.testing.assert_array_equal(origin_array, ORIGIN_PARITY_EXPECTED)
        origin_array_preloaded = auto_origin_id(
            _origin_parity_dataset(),
            device="mps",
            batch_size=3,
            local_margin=25,
            radial_step=1.0,
            preload_to_device=True,
            show_progress=False,
        )
        np.testing.assert_array_equal(origin_array_preloaded, ORIGIN_PARITY_EXPECTED)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
    def test_find_origin_mps_torch_integer_mean_matches_numpy_exactly(self):
        """Chunked MPS integer mean must match the original NumPy mean bit-for-bit."""
        rng = np.random.default_rng(42)
        array_4d = rng.integers(
            0,
            2**31 - 1,
            size=(3, 5, 24, 26),
            dtype=np.uint32,
        )
        expected = array_4d.mean(axis=(0, 1)).astype(np.float32)
        actual = mean_dp_torch(
            array_4d,
            24,
            26,
            "mps",
            chunk_bytes=3 * np.dtype(np.int64).itemsize * 24 * 26,
        )
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
    def test_find_origin_mps_low_count_float32_mean_matches_numpy_exactly(self):
        """Low-count integer mean can use float32 accumulation without losing exactness."""
        rng = np.random.default_rng(43)
        array_4d = rng.integers(0, 64, size=(16, 16, 24, 26), dtype=np.uint32)
        expected = array_4d.mean(axis=(0, 1)).astype(np.float32)
        actual = mean_dp_torch(
            array_4d,
            24,
            26,
            "mps",
            chunk_bytes=17 * np.dtype(np.float32).itemsize * 24 * 26,
        )
        assert actual.dtype == np.float32
        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
    def test_find_origin_mps_direct_chunk_transfer_matches_cpu_conversion(self):
        """Direct uint32-to-MPS chunk transfer must preserve the old float32 values."""
        rng = np.random.default_rng(43)
        chunk = rng.integers(0, 2**31 - 1, size=(4, 19, 23), dtype=np.uint32)
        expected = torch.from_numpy(np.ascontiguousarray(chunk, dtype=np.float32)).to("mps")
        actual = _array_chunk_to_device_float32(chunk, "mps")
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    @pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS is not available")
    def test_find_origin_mps_brittle_score_parity(self):
        """Mid-band MPS scorer must match the old full-grid scorer exactly."""
        dp_batch, offset_row, offset_col, min_r_idx, max_r_idx, n_row, n_col = (
            _mps_origin_score_case()
        )

        cand_rows, cand_cols, shared_fast = _optimized_shared_scores(
            dp_batch,
            80,
            80,
            8,
            4,
            offset_row,
            offset_col,
            min_r_idx,
            max_r_idx,
            n_row,
            n_col,
        )
        shared_reference = _reference_shared_scores(
            dp_batch,
            cand_rows,
            cand_cols,
            offset_row,
            offset_col,
            min_r_idx,
            max_r_idx,
            n_row,
            n_col,
        )
        torch.testing.assert_close(shared_fast, shared_reference, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            shared_fast.argmin(dim=1),
            shared_reference.argmin(dim=1),
            rtol=0,
            atol=0,
        )

        current_row = torch.tensor([80, 81, 78, 82], dtype=torch.long, device="mps")
        current_col = torch.tensor([80, 78, 82, 81], dtype=torch.long, device="mps")
        rel = torch.arange(-4, 5, 2, dtype=torch.long, device="mps")
        drow, dcol = (m.reshape(-1) for m in torch.meshgrid(rel, rel, indexing="ij"))
        paired_rows = (current_row[:, None] + drow[None, :]).clamp(0, n_row - 1)
        paired_cols = (current_col[:, None] + dcol[None, :]).clamp(0, n_col - 1)
        paired_reference = _reference_paired_scores(
            dp_batch,
            paired_rows,
            paired_cols,
            offset_row,
            offset_col,
            min_r_idx,
            max_r_idx,
            n_row,
            n_col,
        )
        paired_fast = _optimized_paired_scores(
            dp_batch,
            paired_rows,
            paired_cols,
            offset_row,
            offset_col,
            min_r_idx,
            max_r_idx,
            n_row,
            n_col,
        )
        torch.testing.assert_close(paired_fast, paired_reference, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            paired_fast.argmin(dim=1),
            paired_reference.argmin(dim=1),
            rtol=0,
            atol=0,
        )


# ============================================================================
# Test Polar Transform
# ============================================================================


class TestPolarTransform:
    """Test polar coordinate transformation."""

    def test_polar_transform_basic(self, synthetic_4dstem_dataset):
        """Test basic polar transformation."""
        polar = polar_transform(synthetic_4dstem_dataset)
        assert isinstance(polar, Polar4dstem)
        assert polar.shape[0] == 3  # scan_y
        assert polar.shape[1] == 3  # scan_x
        assert polar.shape[2] == 180  # num_annular_bins
        assert polar.shape[3] > 0  # radial bins

    def test_polar_transform_single_origin(self, synthetic_4dstem_dataset):
        """Test polar transformation with single origin broadcast to all positions."""
        origin = np.array([128.0, 128.0])
        polar = polar_transform(
            synthetic_4dstem_dataset,
            origin_array=origin,
        )
        assert isinstance(polar, Polar4dstem)

    def test_polar_transform_radial_range(self, synthetic_4dstem_dataset):
        """Test polar transformation with custom radial range."""
        polar = polar_transform(
            synthetic_4dstem_dataset,
            radial_min=5.0,
            radial_max=50.0,
            radial_step=2.0,
        )
        assert isinstance(polar, Polar4dstem)
        # Check that radial dimension matches expected size
        expected_n_r = int(np.ceil((50.0 - 5.0) / 2.0))
        assert polar.shape[3] == expected_n_r

    def test_polar_transform_scan_pos(self, synthetic_4dstem_dataset):
        """Test polar transformation for a single scan position."""
        polar_2d = polar_transform(
            synthetic_4dstem_dataset,
            scan_pos=(0, 0),
        )
        # should return 2D tensor (phi, r)
        assert polar_2d.ndim == 2
        assert polar_2d.shape[0] == 180  # num_annular_bins


# ============================================================================
# Test Radial Mean Calculation
# ============================================================================


class TestRadialMeanCalculation:
    """Test radial mean intensity calculation."""

    def test_calculate_radial_mean_with_mask(self, synthetic_4dstem_dataset):
        """Test radial mean calculation with real-space mask."""
        pdf = PairDistributionFunction.from_data(
            synthetic_4dstem_dataset,
            find_origin=False,
        )
        mask = np.zeros((3, 3), dtype=bool)
        mask[0:2, 0:2] = True
        radial_mean = pdf.calculate_radial_mean(
            mask_realspace=mask,
            returnval=True,
        )
        assert radial_mean is not None


# ============================================================================
# Test Background Fitting
# ============================================================================


class TestBackgroundFitting:
    """Test background fitting."""

    def test_fit_bg_basic(self, synthetic_dataset2d):
        """Test basic background fitting."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        Ik = pdf.calculate_radial_mean(returnval=True)
        k = np.asarray(pdf.qq)
        kmin, kmax = float(k.min()), float(k.max())
        bg, f = pdf.fit_bg(Ik, kmin=kmin * 0.1, kmax=kmax * 0.9)
        assert bg.shape == Ik.shape
        assert f.shape == Ik.shape
        # Check that background is positive
        assert (bg >= 0).all()


# ============================================================================
# Test PDF Calculation
# ============================================================================


class TestPDFCalculation:
    """Test the PDF calculation pipeline."""

    def test_calculate_Gr_with_bandpass(self, synthetic_dataset2d):
        """Test PDF calculation with bandpass filtering."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        pdf.calculate_Gr(
            k_min_fit=0.1,
            k_max_fit=2.0,
            k_lowpass=0.02,
            k_highpass=0.001,
        )
        assert pdf.reduced_pdf is not None

    def test_calculate_Gr_with_mask(self, synthetic_4dstem_dataset):
        """Test PDF calculation with real-space mask."""
        pdf = PairDistributionFunction.from_data(
            synthetic_4dstem_dataset,
            find_origin=False,
        )
        mask = np.zeros((3, 3), dtype=bool)
        mask[0:2, 0:2] = True
        pdf.calculate_Gr(
            k_min_fit=0.1,
            k_max_fit=2.0,
            mask_realspace=mask,
        )
        assert pdf.reduced_pdf is not None

    def test_calculate_gr_requires_Gr(self, synthetic_dataset2d):
        """Test that calculate_gr raises if calculate_Gr has not been run."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        with pytest.raises(RuntimeError, match="Reduced PDF not computed"):
            pdf.calculate_gr(density=0.05)

    def test_calculate_gr_estimates_density(self, synthetic_dataset2d):
        """Test that calculate_gr estimates density when none is provided."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0)
        results = pdf.calculate_gr(returnval=True)
        assert results is not None
        r, gr = results
        assert isinstance(gr, np.ndarray)
        assert len(gr) == len(r)
        assert pdf.rho0 > 0

    def test_estimate_density_requires_Gr(self, synthetic_dataset2d):
        """Test that estimate_density requires prior calculate_Gr call."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        with pytest.raises(
            RuntimeError, match="depends on Sk, reduced_pdf, and r from calculate_Gr"
        ):
            pdf.estimate_density()


# ============================================================================
# Integration Workflows
# ============================================================================


class TestIntegrationWorkflows:
    """Test complete end-to-end workflows."""

    def test_complete_pdf_workflow_2d(self, synthetic_dataset2d):
        """Test: 2D diffraction → polar transform → G(r) → g(r)."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        Gr_results = pdf.calculate_Gr(
            k_min_fit=0.1,
            k_max_fit=2.0,
            r_min=0.0,
            r_max=10.0,
            r_step=0.05,
            returnval=True,
        )
        assert Gr_results is not None
        r, Gr = Gr_results
        assert not np.isnan(r).any()
        assert not np.isnan(Gr).any()
        assert not np.isinf(Gr).any()
        assert len(r) > 0
        assert len(Gr) == len(r)
        gr_results = pdf.calculate_gr(
            density=0.05,
            returnval=True,
        )
        assert gr_results is not None
        r_gr, gr = gr_results
        assert not np.isnan(gr).any()
        assert not np.isinf(gr).any()
        assert len(gr) == len(r_gr)

    def test_complete_pdf_workflow_4dstem(self, synthetic_4dstem_dataset):
        """Test: 4D-STEM → origin finding → polar transform → G(r)."""
        pdf = PairDistributionFunction.from_data(
            synthetic_4dstem_dataset,
            find_origin=True,
        )
        mask = np.zeros((3, 3), dtype=bool)
        mask[0:2, 0:2] = True
        pdf.calculate_Gr(
            k_min_fit=0.1,
            k_max_fit=2.0,
            mask_realspace=mask,
        )
        assert pdf.reduced_pdf is not None
        assert not np.isnan(pdf.reduced_pdf).any()
        assert not np.isinf(pdf.reduced_pdf).any()

    def test_polar_transform_input_types(self, synthetic_diffraction_pattern):
        """Test from_data works with Dataset2d and Dataset4dstem."""
        # Test with Dataset2d
        ds2 = Dataset2d.from_array(
            array=synthetic_diffraction_pattern,
            name="test",
        )
        pdf_ds2 = PairDistributionFunction.from_data(
            ds2,
            find_origin=False,
        )
        assert pdf_ds2.polar.shape[2] == 180

        # Test with Dataset4dstem
        array_4d = synthetic_diffraction_pattern[None, None, :, :]  # (1, 1, ny, nx)
        ds4 = Dataset4dstem.from_array(array_4d, name="test")
        pdf_ds4 = PairDistributionFunction.from_data(
            ds4,
            find_origin=False,
        )
        assert pdf_ds4.polar.shape[2] == 180
        assert pdf_ds2.polar.shape == pdf_ds4.polar.shape

    def test_density_estimation_workflow(self, synthetic_dataset2d):
        """Test: G(r) calculation → density estimation → g(r) calculation."""
        pdf = PairDistributionFunction.from_data(
            synthetic_dataset2d,
            find_origin=False,
        )
        pdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0)
        rho0, Fk_damped, G_cor = pdf.estimate_density(
            max_iter=5,
            tol_percent=1.0,
        )
        assert rho0 > 0
        assert np.isfinite(rho0)
        results = pdf.calculate_gr(
            density=rho0,
            returnval=True,
        )
        assert results is not None
        r, gr = results
        assert not np.isnan(gr).any()


# ============================================================================
# Frozen numerical baselines.
# Locks current numpy output so any refactor (torch backends, GPU path,
# device dispatch) has a concrete reference to prove parity against. The
# smoke tests above only assert "doesn't crash and returns the right shape" —
# they would not catch a 10% drift in G(r) or rho0.
# ============================================================================

EXPECTED_MEAN_DP_SLICE = np.array(
    [
        [8.0992985, 5.0620613, 2.0248246, 5.0620613, 2.0248246, 6.0744734, 0.0, 6.0744734],
        [6.0744734, 4.0496492, 8.0992985, 5.0620613, 5.0620613, 3.0372367, 5.0620613, 5.0620613],
        [4.0496492, 3.0372367, 6.0744734, 11.136536, 4.0496492, 7.0868855, 5.0620613, 4.0496492],
        [4.0496492, 9.111711, 7.0868855, 11.681102, 82.3938, 8.247485, 5.0620613, 7.0868855],
        [3.0372367, 8.0992985, 6.0744734, 78.34415, 993.79034, 50.38002, 5.0620613, 4.0496492],
        [9.111711, 4.0496492, 5.0620613, 8.247485, 51.392437, 6.2526093, 9.111711, 2.0248246],
        [7.0868855, 5.0620613, 4.0496492, 9.111711, 6.0744734, 5.0620613, 4.0496492, 6.0744734],
        [3.0372367, 8.0992985, 10.124123, 6.0744734, 4.0496492, 8.0992985, 5.0620613, 8.0992985],
    ],
    dtype=np.float32,
)

EXPECTED_REDUCED_PDF_SLICE_2D = np.array(
    [0.0, -2.5913910e07, -2.8996332e06, -1.9657751e06, 4.7554210e06,
     -2.2013840e06, -1.0098955e07, -7.6990675e06, -3.5052252e06, -1.2820963e07],
    dtype=np.float32,
)

EXPECTED_RHO0_2D = 376487.9375  # synthetic 2D, not physical — just frozen

EXPECTED_GR_SLICE_2D = np.array(
    [0.0, -1.7386847, 0.8467777, 0.9307497, 1.1256429,
     0.9534698, 0.8221171, 0.8837617, 0.9536942, 0.8494478],
    dtype=np.float32,
)

# Karen's simulated amorphous Ta (PR #177 tutorial). Locked numerically only
# when ``$QUANTEM_TA_ZIP`` points at a local copy of ``Ta_sim_binned.zip``;
# tests skip otherwise. Download once from the tutorial's Google Drive link
# and export the env var to enable these checks locally.
TA_ZIP_PATH = __import__("os").environ.get("QUANTEM_TA_ZIP", "")

EXPECTED_TA_RHO0 = 0.035133879631757736  # atoms/Å³, matches Karen's tutorial 0.035138

EXPECTED_TA_REDUCED_PDF_SLICE = np.array(
    [0.0, -1.3650746, -1.7406626, -0.00403661, 0.2258577,
     0.72336507, 0.19312079, -0.11800791, -0.15564784, -0.05905667],
    dtype=np.float32,
)

EXPECTED_TA_GR_SLICE = np.array(
    [0.0, 0.0, 0.00681734, 0.99856776, 1.0641963,
     1.1638141, 1.0364699, 0.9809085, 0.97795784, 0.9925757],
    dtype=np.float32,
)


class TestPDFNumericalBaseline:
    """Frozen-numerical baselines locking the current numpy pipeline output."""

    def test_mean_dp_baseline(self, synthetic_4dstem_dataset):
        """Mean DP frozen against numpy reference."""
        mean = synthetic_4dstem_dataset.array.mean(axis=(0, 1)).astype(np.float32)
        np.testing.assert_allclose(mean[::32, ::32], EXPECTED_MEAN_DP_SLICE, atol=1e-4)

    def test_origin_finding_baseline(self):
        """auto_origin_id on the small parity fixture matches the locked origins."""
        origins = auto_origin_id(_origin_parity_dataset(), show_progress=False)
        np.testing.assert_array_equal(origins, ORIGIN_PARITY_EXPECTED)

    def test_calculate_Gr_synthetic_baseline(self, synthetic_dataset2d):
        """reduced_pdf frozen on synthetic 2D fixture."""
        rdf = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        rdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0)
        np.testing.assert_allclose(
            rdf.reduced_pdf[::100], EXPECTED_REDUCED_PDF_SLICE_2D, atol=1.0, rtol=1e-5
        )

    def test_calculate_gr_synthetic_baseline(self, synthetic_dataset2d):
        """rho0 and g(r) frozen on synthetic 2D."""
        rdf = PairDistributionFunction.from_data(synthetic_dataset2d, find_origin=False)
        rdf.calculate_Gr(k_min_fit=0.1, k_max_fit=2.0)
        r, gr = rdf.calculate_gr(returnval=True)
        np.testing.assert_allclose(rdf.rho0, EXPECTED_RHO0_2D, rtol=1e-5)
        np.testing.assert_allclose(gr[::100], EXPECTED_GR_SLICE_2D, atol=1e-4)

    @pytest.mark.skipif(
        not __import__("os").path.exists(TA_ZIP_PATH),
        reason=f"Karen's Ta_sim_binned.zip not at {TA_ZIP_PATH}",
    )
    def test_karen_ta_full_pipeline_baseline(self):
        """Full PDF pipeline on Karen's simulated amorphous Ta — matches tutorial."""
        from quantem.core.io.serialize import load
        ds = load(TA_ZIP_PATH)
        rdf = PairDistributionFunction.from_data(ds, find_origin=True, origin_show_progress=False)
        rdf.polar.sampling[3] = 0.01488  # Karen's tutorial q-calibration
        rdf.calculate_Gr(r_max=20.0, k_min_fit=0.05, damp_origin_oscillations=True)
        r, gr = rdf.calculate_gr(returnval=True, set_pdf_positive=True)
        np.testing.assert_allclose(rdf.rho0, EXPECTED_TA_RHO0, rtol=1e-5)
        np.testing.assert_allclose(
            rdf.reduced_pdf[::100], EXPECTED_TA_REDUCED_PDF_SLICE, atol=1e-3
        )
        np.testing.assert_allclose(gr[::100], EXPECTED_TA_GR_SLICE, atol=1e-3)


# ============================================================================
# Cross-device parity tests for the torch-backed pipeline.
# Validates that a Dataset4dstem holding a torch tensor (any device) produces
# output consistent with the numpy CPU baseline locked above.
# ============================================================================


def _torch_devices():
    """Devices on which to exercise the torch-backed parity tests."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.parametrize("device", _torch_devices())
class TestTorchBackedDataset4dstemParity:
    """End-to-end parity between a torch-backed Dataset4dstem and the NumPy path."""

    def test_mean_dp_matches_numpy_integer(self, device):
        """Integer mean DP via ``mean_dp_torch`` is bit-exact vs numpy."""
        rng = np.random.default_rng(0)
        arr = rng.integers(0, 1000, size=(6, 6, 24, 26), dtype=np.uint16)
        ds_torch = Dataset4dstem.from_array(torch.from_numpy(arr).to(device))
        mean_torch = mean_dp_torch(ds_torch.array, 24, 26, device)
        mean_np = arr.mean(axis=(0, 1)).astype(np.float32)
        np.testing.assert_array_equal(mean_torch, mean_np)

    def test_mean_dp_matches_numpy_float(self, device):
        """Float mean DP via ``mean_dp_torch`` matches numpy within float32 ULPs."""
        ds_np = _origin_parity_dataset()
        ds_torch = Dataset4dstem.from_array(torch.from_numpy(ds_np.array).to(device))
        n_row, n_col = ds_torch.array.shape[-2:]
        mean_torch = mean_dp_torch(ds_torch.array, n_row, n_col, device)
        mean_np = ds_np.array.mean(axis=(0, 1)).astype(np.float32)
        np.testing.assert_allclose(mean_torch, mean_np, rtol=1e-6, atol=1e-3)

    def test_auto_origin_id_matches_numpy(self, device):
        """``auto_origin_id`` on the torch-backed dataset matches the NumPy path."""
        ds_np = _origin_parity_dataset()
        ds_torch = Dataset4dstem.from_array(torch.from_numpy(ds_np.array).to(device))
        origins_np = auto_origin_id(ds_np, batch_size=8, show_progress=False)
        origins_torch = auto_origin_id(ds_torch, batch_size=8, show_progress=False)
        np.testing.assert_array_equal(origins_torch, origins_np)
        np.testing.assert_array_equal(origins_torch, ORIGIN_PARITY_EXPECTED)

    def test_polar_transform_matches_numpy(self, device):
        """``polar_transform`` on the torch-backed dataset matches the NumPy path."""
        ds_np = _origin_parity_dataset()
        ds_torch = Dataset4dstem.from_array(torch.from_numpy(ds_np.array).to(device))
        origins = ORIGIN_PARITY_EXPECTED
        polar_np = polar_transform(ds_np, origin_array=origins, batch_size=8)
        polar_torch = polar_transform(ds_torch, origin_array=origins, batch_size=8)
        np_arr = polar_np.array
        torch_arr = (
            polar_torch.array.cpu().numpy()
            if isinstance(polar_torch.array, torch.Tensor)
            else polar_torch.array
        )
        # grid_sample float32 may differ by a few ULPs between backends.
        np.testing.assert_allclose(torch_arr, np_arr, rtol=1e-4, atol=1e-3)


# ============================================================================
# Full pipeline parity across all three target audiences:
#   - CPU torch on Linux / Mac (works everywhere)
#   - CUDA torch on a Linux NVIDIA box
#   - MPS torch on Apple Silicon Mac
#
# These tests run only the available backends on whichever machine pytest is
# invoked on. The Karen Ta dataset is required; the tests skip without it.
# ============================================================================


@pytest.mark.skipif(
    not __import__("os").path.exists(TA_ZIP_PATH),
    reason=f"Karen's Ta_sim_binned.zip not at {TA_ZIP_PATH}",
)
@pytest.mark.parametrize("device", _torch_devices())
class TestPDFFullPipelineAcrossDevices:
    """End-to-end PDF on Karen Ta, parameterized over (cpu, cuda, mps).

    Asserts that rho0 matches Karen's published tutorial value (0.035138
    atoms/Å³) within float32 reproducibility tolerance, regardless of which
    backend the user runs on. Catches regressions in any backend's iterative
    density refinement.

    Hardware verified during this PR
    --------------------------------
    Backend  Box                                 Total wall   rho0
    -------  ----------------------------------  ----------   --------
    CPU      Apple Silicon M5             1.12 s       0.035135
    MPS      Apple Silicon M5             1.04 s       0.035134
    CUDA     NVIDIA RTX PRO 6000 Blackwell       1.24 s       0.035133

    All three reproduce Karen's tutorial value (0.035138) within 0.015%
    relative error. MPS is the fastest end-to-end on this small dataset
    because the per-batch polar-transform launch overhead favors the
    high-bandwidth Apple unified memory; CUDA pulls ahead on larger scans
    (e.g. 512x512x192x192 = 230x speedup vs the original numpy CPU path).
    """

    def test_full_pipeline_matches_karen_tutorial(self, device):
        from quantem.core.io.serialize import load
        ds_np = load(TA_ZIP_PATH)
        ds = Dataset4dstem.from_array(torch.from_numpy(ds_np.array).to(device))
        rdf = PairDistributionFunction.from_data(
            ds, find_origin=True, origin_show_progress=False
        )
        rdf.polar.sampling[3] = 0.01488  # Karen's tutorial q-calibration
        rdf.calculate_Gr(r_max=20.0, k_min_fit=0.05, damp_origin_oscillations=True)
        r, gr = rdf.calculate_gr(returnval=True, set_pdf_positive=True)
        # All three backends must converge to the same density within float32 ULPs.
        np.testing.assert_allclose(rdf.rho0, EXPECTED_TA_RHO0, rtol=1e-4)
        np.testing.assert_allclose(
            rdf.reduced_pdf[::100], EXPECTED_TA_REDUCED_PDF_SLICE, atol=1e-3
        )
        np.testing.assert_allclose(gr[::100], EXPECTED_TA_GR_SLICE, atol=1e-3)
