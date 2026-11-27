"""Serpentine and interlaced (2x2 / 4x4) STEM scan drift correction.

Supports single-pass bidirectional serpentine scans and multi-pass interlaced
serpentine scans (``serpentine_2x2``, ``serpentine_4x4``, and rotated variants)
using:

1. Zero-slope-bias symmetric midpoint bidirectional snake-lag estimation
   (``0.5 * (line[2k] + line[2k+2])`` vs ``line[2k+1]``) with anti-aliasing
   so tilted features and periodic lattices never produce false lag bias.
2. Exact sub-grid offset compensation between interleaved passes
   (e.g., ``(0, 0), (0, 1), (1, 1), (1, 0)`` for ``2x2``).
3. Multi-pass orthogonal/antiparallel joint drift correction via
   :class:`DriftCorrection` (``0°, -90°, 180°, +90°`` for ``serpentine_2x2``),
   re-interleaving corrected probe coordinates onto the full ``(H, W, 2)`` grid.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter, shift as ndi_shift

SERPENTINE_SINGLE_PATTERNS = ("serpentine",)
SERPENTINE_INTERLEAVED_PATTERNS: dict[str, int] = {
    "serpentine_2x2": 2,
    "serpentine_4x4": 4,
    "serpentine_2x2_rotated": 2,
    "serpentine_4x4_rotated": 4,
}
SERPENTINE_ROTATED_PATTERNS = ("serpentine_2x2_rotated", "serpentine_4x4_rotated")
ALL_SERPENTINE_PATTERNS = SERPENTINE_SINGLE_PATTERNS + tuple(SERPENTINE_INTERLEAVED_PATTERNS)


def is_serpentine_pattern(pattern: str | None) -> bool:
    """Return True if ``pattern`` is a single-pass or interleaved serpentine pattern."""
    return isinstance(pattern, str) and pattern.strip().lower() in ALL_SERPENTINE_PATTERNS


def is_interleaved_pattern(pattern: str | None) -> bool:
    """Return True if ``pattern`` is a multi-pass interlaced serpentine scan."""
    return isinstance(pattern, str) and pattern.strip().lower() in SERPENTINE_INTERLEAVED_PATTERNS


def interlace_stride(pattern: str | None) -> int:
    """Return the sub-grid stride ``k`` (2 for 2x2, 4 for 4x4, 1 otherwise)."""
    if not isinstance(pattern, str):
        return 1
    return SERPENTINE_INTERLEAVED_PATTERNS.get(pattern.strip().lower(), 1)


def interlace_offsets(k: int) -> list[tuple[int, int]]:
    """Return the exact ``(dr, dc)`` pass offsets in full-grid pixels.

    Matches the ARINA / Spectra interlaced serpentine visit schedule:
    - ``k = 2``: ``[(0, 0), (0, 1), (1, 1), (1, 0)]`` (half-step square)
    - ``k = 4``: coarse-to-fine space-filling order over ``4 x 4`` offsets
    """
    k = int(k)
    if k == 2:
        return [(0, 0), (0, 1), (1, 1), (1, 0)]
    if k == 4:
        # Coarse-to-fine bit-reversed 2D Morton / quadrant order
        order_1d = [0, 2, 1, 3]
        offsets: list[tuple[int, int]] = []
        for r_base in (0, 1):
            for c_base in (0, 1):
                for dr in (0, 2):
                    for dc in (0, 2):
                        offsets.append((dr + r_base, dc + c_base))
        # Deduplicate while preserving order
        seen = set()
        out = []
        for rc in offsets:
            if rc not in seen:
                seen.add(rc)
                out.append(rc)
        return out
    return [(r, c) for r in range(k) for c in range(k)]


def interlace_pass_specs(rows: int, cols: int, scan_pattern: str) -> list[dict[str, Any]]:
    """Compute exact pass offsets, fast axes, and DriftCorrection angles for a pattern.

    Follows nearest-corner pass continuation:
    - For ``serpentine_2x2`` on even ``(rows/2, cols/2)``:
      Pass 0 ``(0, 0)``: fast=row (axis 1), slow=+row  -> ``0°``,   ``rot90_k=0``
      Pass 1 ``(0, 1)``: fast=col (axis 0), slow=+col  -> ``-90°``, ``rot90_k=-1``
      Pass 2 ``(1, 1)``: fast=row (axis 1), slow=-row  -> ``180°``, ``rot90_k=2``
      Pass 3 ``(1, 0)``: fast=col (axis 0), slow=-col  -> ``+90°``, ``rot90_k=+1``
    """
    pat = scan_pattern.strip().lower()
    k = interlace_stride(pat)
    if k <= 1:
        return [{
            "index": 0,
            "offset": (0, 0),
            "fast_axis": 1,
            "angle_deg": 0.0,
            "rot90_k": 0,
            "stride": 1,
        }]

    n_rows, n_cols = rows // k, cols // k
    rotated = pat in SERPENTINE_ROTATED_PATTERNS
    offsets = interlace_offsets(k)
    specs: list[dict[str, Any]] = []
    last: tuple[int, int] | None = None

    for idx, (dr, dc) in enumerate(offsets):
        r0, r1 = dr, dr + k * (n_rows - 1)
        c0, c1 = dc, dc + k * (n_cols - 1)
        corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
        if last is None:
            start = corners[0]
        else:
            dists = [(cr - last[0]) ** 2 + (cc - last[1]) ** 2 for cr, cc in corners]
            start = corners[int(np.argmin(dists))]

        row_asc = start[0] == r0
        col_asc = start[1] == c0

        if rotated:
            angle_deg = float((180.0 * idx / (k * k)) % 180.0)
            # Nearest cardinal fast axis for de-combing
            fast_axis = 1 if (angle_deg < 45.0 or angle_deg >= 135.0) else 0
            rot90_k = int(round(-angle_deg / 90.0)) % 4
            # Approximate end corner for nearest-corner chaining
            end = (r1 if row_asc else r0, c1 if col_asc else c0)
        elif idx % 2 == 0:
            # Horizontal fast axis (along rows)
            fast_axis = 1
            if row_asc:
                angle_deg, rot90_k = 0.0, 0
            else:
                angle_deg, rot90_k = 180.0, 2
            end_r = r1 if row_asc else r0
            # Even n_rows -> last row runs backward back to start col
            end_c = start[1] if (n_rows % 2 == 0) else (c1 if col_asc else c0)
            end = (end_r, end_c)
        else:
            # Vertical fast axis (along cols)
            fast_axis = 0
            if col_asc:
                # Slow axis left -> right (+col): matches DriftCorrection angle = -90°
                angle_deg, rot90_k = -90.0, -1
            else:
                # Slow axis right -> left (-col): matches DriftCorrection angle = +90°
                angle_deg, rot90_k = 90.0, 1
            end_c = c1 if col_asc else c0
            end_r = start[0] if (n_cols % 2 == 0) else (r1 if row_asc else r0)
            end = (end_r, end_c)

        specs.append({
            "index": idx,
            "offset": (int(dr), int(dc)),
            "fast_axis": int(fast_axis),
            "angle_deg": float(angle_deg),
            "rot90_k": int(rot90_k),
            "stride": k,
        })
        last = end
    return specs


def _estimate_min_period_px(img: np.ndarray) -> float | None:
    """Fast streak-masked FFT lattice period detector to prevent snake-lag aliasing."""
    a = np.asarray(img, dtype=np.float32)
    h, w = a.shape
    if min(h, w) < 32:
        return None
    a = a - float(np.mean(a))
    wy = np.hanning(h)[:, None]
    wx = np.hanning(w)[None, :]
    pwr = np.abs(np.fft.fftshift(np.fft.fft2(a * wy * wx))) ** 2
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    ky = (yy - cy) / float(h)
    kx = (xx - cx) / float(w)
    kr = np.hypot(ky, kx)
    # Mask DC and horizontal/vertical axial streaks
    mask = (kr >= 1.0 / min(h, w) * 6.0) & (kr <= 0.45) & (np.abs(yy - cy) > 2) & (np.abs(xx - cx) > 2)
    if not np.any(mask):
        return None
    vals = pwr[mask]
    med = float(np.median(vals))
    if med <= 0:
        return None
    peak_idx = int(np.argmax(vals))
    peak_val = float(vals[peak_idx])
    if peak_val < 35.0 * med:
        return None
    k_peak = float(kr[mask][peak_idx])
    if k_peak <= 0:
        return None
    return 1.0 / k_peak


def estimate_snake_lag(
    img: np.ndarray,
    fast_axis: int = 1,
    max_lag: int = 6,
    lattice_period_px: float | None = None,
) -> dict[str, Any]:
    """Estimate bidirectional even/odd scanline lag with zero tilted-feature bias.

    Compares symmetric midpoint-interpolated even lines ``0.5 * (L[2k] + L[2k+2])``
    against odd lines ``L[2k+1]``. Because both sit at identical slow-axis coordinate
    ``2k + 1``, angled interfaces and tilted atomic rows introduce zero slope bias.
    When a periodic lattice is present, the search window is bounded to ``0.42 * P_min``
    so de-combing can never alias by a full unit cell.
    """
    a = np.asarray(img, dtype=np.float32)
    if fast_axis == 0:
        a = a.T
    if a.shape[0] < 8 or a.shape[1] < 16:
        return {"lag_px": 0.0, "half_lag_px": 0.0, "corr_before": 1.0, "corr_after": 1.0, "applied": False}

    pmin = lattice_period_px if lattice_period_px is not None else _estimate_min_period_px(a)
    eff_max = min(int(max_lag), max(1, int(math.floor(0.42 * pmin)))) if (pmin is not None and pmin >= 2.5) else int(max_lag)

    def sym_ncc(arr: np.ndarray, lag: int) -> float:
        e_mid = 0.5 * (arr[0:-2:2] + arr[2::2])
        o_mid = arr[1 : 1 + 2 * e_mid.shape[0] : 2]
        if lag == 0:
            ea, oa = e_mid, o_mid
        elif lag > 0:
            ea, oa = e_mid[:, lag:], o_mid[:, :-lag]
        else:
            ea, oa = e_mid[:, :lag], o_mid[:, -lag:]
        ea = ea - float(ea.mean())
        oa = oa - float(oa.mean())
        denom = float(np.sqrt(float((ea * ea).sum()) * float((oa * oa).sum()))) + 1e-12
        return float((ea * oa).sum() / denom)

    raw_scores = {lag: sym_ncc(a, lag) for lag in range(-eff_max, eff_max + 1)}
    c_zero = raw_scores[0]
    best_int = max(raw_scores, key=raw_scores.get)
    # Require a genuine gain over lag=0 to move off 0
    if best_int != 0 and raw_scores[best_int] <= c_zero + 0.01:
        best_int = 0

    c_m = sym_ncc(a, best_int - 1) if best_int - 1 >= -eff_max else raw_scores[best_int]
    c_0 = raw_scores[best_int]
    c_p = sym_ncc(a, best_int + 1) if best_int + 1 <= eff_max else raw_scores[best_int]
    denom = c_m - 2.0 * c_0 + c_p
    sub = float(np.clip(0.5 * (c_m - c_p) / denom, -0.5, 0.5)) if (abs(denom) > 1e-12 and denom < 0.0) else 0.0
    lag = float(best_int + sub)

    if abs(lag) < 0.06:
        return {"lag_px": 0.0, "half_lag_px": 0.0, "corr_before": round(c_zero, 4), "corr_after": round(c_zero, 4), "applied": False}

    shifted = decomb_serpentine(a, lag, fast_axis=1)
    c_after = sym_ncc(shifted, 0)
    if c_after <= c_zero + 0.002:
        return {"lag_px": 0.0, "half_lag_px": 0.0, "corr_before": round(c_zero, 4), "corr_after": round(c_zero, 4), "applied": False}

    return {
        "lag_px": round(lag, 3),
        "half_lag_px": round(0.5 * lag, 3),
        "corr_before": round(c_zero, 4),
        "corr_after": round(c_after, 4),
        "applied": True,
    }


def decomb_serpentine(img: np.ndarray, lag_px: float, fast_axis: int = 1) -> np.ndarray:
    """Remove bidirectional snake lag symmetrically (``-lag/2`` even, ``+lag/2`` odd)."""
    a = np.asarray(img, dtype=np.float32)
    if abs(lag_px) < 0.05:
        return a.copy()
    if fast_axis == 0:
        a = a.T
    out = np.empty_like(a)
    half = 0.5 * float(lag_px)
    out[0::2] = ndi_shift(a[0::2], shift=(0.0, -half), order=3, mode="nearest")
    out[1::2] = ndi_shift(a[1::2], shift=(0.0, +half), order=3, mode="nearest")
    return out.T.copy() if fast_axis == 0 else out


def split_interleaved_passes(
    img: np.ndarray,
    scan_pattern: str,
    *,
    decomb: bool = True,
    compensate_subgrid: bool = True,
    fixed_lags: list[float] | None = None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    """Split a full-grid interlaced scan into aligned sub-passes with metadata.

    Parameters
    ----------
    img : ndarray, shape (H, W)
        Full-grid image (reordered from visits onto the ``(H, W)`` scan grid).
    scan_pattern : str
        e.g. ``"serpentine_2x2"``, ``"serpentine_4x4"``, ``"serpentine_2x2_rotated"``.
    decomb : bool, default True
        Estimate and remove bidirectional snake lag per pass.
    compensate_subgrid : bool, default True
        Shift each pass by its exact geometric sub-grid offset ``(+dr/k, +dc/k)``
        so all passes share the exact ``(0, 0)`` coordinate origin before drift fitting.
    fixed_lags : list[float] or None, optional
        If supplied (e.g., applying DF-estimated lags to Direct Ptycho), uses these
        exact per-pass lags instead of re-estimating.

    Returns
    -------
    passes : list[ndarray]
        Sub-pass arrays of shape ``(H/k, W/k)`` in sample coordinates (not yet rotated
        into scan frame).
    specs : list[dict]
        Per-pass metadata (``offset``, ``fast_axis``, ``angle_deg``, ``rot90_k``,
        ``stride``, ``snake_lag``).
    """
    arr = np.asarray(img, dtype=np.float32)
    h, w = arr.shape
    specs = interlace_pass_specs(h, w, scan_pattern)
    passes: list[np.ndarray] = []

    for i, sp in enumerate(specs):
        k = sp["stride"]
        dr, dc = sp["offset"]
        sub = arr[dr::k, dc::k].copy()
        ax = sp["fast_axis"]
        if fixed_lags is not None and i < len(fixed_lags):
            lag_val = float(fixed_lags[i])
            lag_info = {"lag_px": round(lag_val, 3), "half_lag_px": round(0.5 * lag_val, 3), "applied": abs(lag_val) >= 0.05}
        elif decomb:
            lag_info = estimate_snake_lag(sub, fast_axis=ax)
            lag_val = float(lag_info["lag_px"])
        else:
            lag_val = 0.0
            lag_info = {"lag_px": 0.0, "half_lag_px": 0.0, "applied": False}

        if abs(lag_val) >= 0.05:
            sub = decomb_serpentine(sub, lag_val, fast_axis=ax)
        if compensate_subgrid and (dr != 0 or dc != 0):
            # Pass(i, j) samples full grid at (k*i + dr, k*j + dc) = k*(i + dr/k, j + dc/k);
            # shifting by (+dr/k, +dc/k) resamples onto the shared (k*i, k*j) reference grid.
            sub = ndi_shift(sub, shift=(float(dr) / k, float(dc) / k), order=3, mode="nearest")

        sp_out = dict(sp)
        sp_out["snake_lag"] = lag_info
        specs[i] = sp_out
        passes.append(sub.astype(np.float32, copy=False))

    return passes, specs


def reinterleave_probe_positions(
    dc: Any,
    specs: list[dict[str, Any]],
    full_shape: tuple[int, int],
    *,
    corrected: bool = True,
) -> dict[str, Any]:
    """Re-interleave per-pass DriftCorrection probe positions onto full ``(H, W, 2)``.

    Accounts for stride ``k``, geometric offset ``(dr, dc)``, and bidirectional
    snake lag ``±lag/2`` so exported probe coordinates are exact for iterative
    ptychography.
    """
    h_full, w_full = int(full_shape[0]), int(full_shape[1])
    yy, xx = np.mgrid[0:h_full, 0:w_full].astype(np.float32)
    nominal_full = np.stack([yy, xx], axis=-1)
    corrected_full = nominal_full.copy()
    per_pass_det_j: list[float] = []

    for i, sp in enumerate(specs):
        k = int(sp["stride"])
        dr, dc_o = sp["offset"]
        rot90_k = int(sp.get("rot90_k", 0))
        pos_pass = np.asarray(
            dc.probe_positions(image_index=i, corrected=corrected, strip_padding=True, plot=False),
            dtype=np.float32,
        )
        # Un-rotate array indexing from rotated scan frame back to sample (pr, pc) order!
        if rot90_k % 4 != 0:
            pos_pass = np.ascontiguousarray(np.rot90(pos_pass, k=-rot90_k))
        # Signed Jacobian determinant on pass grid (positive nominal handedness for all passes)
        if pos_pass.shape[0] >= 2 and pos_pass.shape[1] >= 2:
            pr = np.diff(pos_pass[:, :-1, 0], axis=0)
            pc = np.diff(pos_pass[:, :-1, 1], axis=0)
            qr = np.diff(pos_pass[:-1, :, 0], axis=1)
            qc = np.diff(pos_pass[:-1, :, 1], axis=1)
            det = pr * qc - pc * qr
            per_pass_det_j.append(round(float(np.min(det)), 4))
        else:
            per_pass_det_j.append(1.0)

        # Map pass coordinates -> full-grid coordinates and restore exact sub-grid offset + snake lag
        pos_full = pos_pass * float(k)
        pos_full[..., 0] += float(dr)
        pos_full[..., 1] += float(dc_o)

        lag_px = float((sp.get("snake_lag") or {}).get("lag_px", 0.0))
        if corrected and abs(lag_px) >= 0.05:
            # decomb_serpentine applies ndi_shift(..., shift=-lag/2) on even lines (out[x]=in[x+lag/2]),
            # so raw pixel x actually sampled physical coordinate x - lag/2 (and odd lines x + lag/2).
            half_full = 0.5 * lag_px * float(k)
            if int(sp["fast_axis"]) == 1:
                pos_full[0::2, :, 1] -= half_full
                pos_full[1::2, :, 1] += half_full
            else:
                pos_full[:, 0::2, 0] -= half_full
                pos_full[:, 1::2, 0] += half_full

        corrected_full[dr::k, dc_o::k] = pos_full

    return {
        "nominal": nominal_full,
        "corrected": corrected_full,
        "per_pass_min_det_j": per_pass_det_j,
        "min_det_j": float(min(per_pass_det_j)) if per_pass_det_j else 1.0,
    }
