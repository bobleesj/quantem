"""MacBook 4D-STEM viewer (Show4DSTEM_MACBOOK) — MPS, raw Metal, no torch.

The interactive widget: full-resolution diffraction-pattern display + live BF/DF/
ADF on a bin2 sidecar. The compute reductions live in
quantem.widget.kernels.compute.mps (MetalVirtualImage / ChunkedFrames); this module
is UI only and imports them. See docs/dev-notes/2026-06-01-kernels-backend-architecture.md.
"""
from __future__ import annotations

import gc
import time
import threading
import numpy as np
import traitlets

from quantem.widget.show4dstem import Show4DSTEM
from quantem.widget.kernels.compute.mps import (
    ChunkedFrames,
    MetalVirtualImage,
    MultiChunkedFrames,
    _DEFAULT_COMPACT_TARGET_BYTES,
    _bin_mask,
    _upsample_bin_dp,
)
from quantem.widget.kernels.io.mps import (
    load_mps_4dstem,
    clear_mps_cache,
    MPSChunked4DSTEM,
)

# Idle delay (s) before the background radial-interaction builder polls again.
_RADIAL_INTERACTION_IDLE_DELAY = 0.75


class Show4DSTEMMPS(Show4DSTEM):
    """Show4DSTEM over a no-bin uint16 stack with raw-Metal BF/DF (no torch)."""

    fast_interaction = traitlets.Bool(False).tag(sync=True)
    fast_interaction_ready = traitlets.Bool(False).tag(sync=True)
    fast_interaction_building = traitlets.Bool(False).tag(sync=True)
    radial_interaction_ready = traitlets.Bool(False).tag(sync=True)
    radial_interaction_building = traitlets.Bool(False).tag(sync=True)

    def __init__(
        self,
        *args,
        fast_interaction: bool = False,
        fast_interaction_verbose: bool = True,
        fast_interaction_async: bool = False,
        full_resolution_interaction: bool = False,
        auto_detect_frames: int | None = 64,
        initial_preset: str | None = "BF",
        **kwargs,
    ):
        verbose = bool(kwargs.pop("verbose", True))
        if full_resolution_interaction:
            raise ValueError(
                "full_resolution_interaction has been disabled on the MacBook "
                "viewer path. Use load(...) plus show_4dstem_mps(...) for the "
                "supported real-time fast path."
            )
        self._fast_interaction_verbose = bool(fast_interaction_verbose)
        self._fast_interaction_async = bool(fast_interaction_async)
        self._fast_interaction_thread = None
        self._fast_interaction_error = None
        self._radial_interaction_thread = None
        self._radial_interaction_error = None
        self._radial_interaction_request = 0
        self._radial_interaction_pending_center = None
        self.auto_detect_frames = auto_detect_frames
        self._suppress_fast_interaction_observer = False
        self._mps_initializing = True
        kwargs.setdefault("precompute_virtual_images", False)
        t0 = time.perf_counter()
        try:
            # Suppress the inherited torch-centric "to cpu" line. This wrapper
            # uses CPU torch tensors only for tiny compatibility masks; raw data
            # and virtual images stay on Metal buffers/kernels.
            super().__init__(*args, verbose=False, **kwargs)
        finally:
            self._mps_initializing = False
        self._det_row_coords_np = np.arange(self.det_rows, dtype=np.float32)[:, None]
        self._det_col_coords_np = np.arange(self.det_cols, dtype=np.float32)[None, :]
        self._wire_multi_dataset()
        self.observe(
            self._on_fast_interaction_change,
            names=["fast_interaction"],
        )
        pre_binned_fast = (
            isinstance(self._data, ChunkedFrames)
            and int(getattr(self._data, "det_bin", 1)) > 1
        )
        fused_fast = (
            isinstance(self._data, ChunkedFrames)
            and getattr(self._data, "fast_vi", None) is not None
        )
        if pre_binned_fast:
            fast_interaction = False
            self.fast_interaction_ready = True
        elif fused_fast:
            self.fast_interaction_ready = True
        if initial_preset is not None:
            self._mps_initializing = True
            try:
                self.apply_preset(initial_preset)
            finally:
                self._mps_initializing = False
        if full_resolution_interaction and isinstance(self._data, ChunkedFrames):
            if self._data.vi.row_prefix_enabled:
                self._data.vi._warm_row_prefix_numba()
            else:
                self._data.vi.enable_row_prefix(verbose=verbose)
                _drop_cached_decompressor()
                gc.collect()
        if fast_interaction:
            self.set_fast_interaction(True, wait=not self._fast_interaction_async)
            if fused_fast:
                self._cache_fast_presets()
        else:
            self._clear_virtual_image_caches()
            self._compute_virtual_image_from_roi()
            if full_resolution_interaction:
                self._start_radial_interaction_background()
        if verbose:
            det_bin = int(getattr(self._data, "det_bin", 1)) if isinstance(
                self._data, ChunkedFrames
            ) else 1
            fb = int(getattr(self._data, "fast_bin", 2)) if isinstance(
                self._data, ChunkedFrames) else 2
            mode = (
                f"fast detector-bin{det_bin}"
                if det_bin > 1 else
                "full 192x192 exact row-prefix"
                if full_resolution_interaction else
                f"fast bin{fb} ready" if fast_interaction and self.fast_interaction_ready else
                f"fast bin{fb} async" if fast_interaction and fast_interaction_async else
                f"fast bin{fb}" if fast_interaction else "full 192x192 exact"
            )
            shape = f"{self.shape_rows}x{self.shape_cols}x{self.det_rows}x{self.det_cols}"
            print(
                f"Ready MPS viewer in {time.perf_counter() - t0:.2f}s "
                f"({shape}, Raw Metal, {mode})"
            )

    def auto_detect_center(self, update_roi: bool = True):
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return super().auto_detect_center(update_roi)
        sample = self.auto_detect_frames
        if sample is not None and int(sample) > 0 and int(sample) < data._n:
            sample = int(sample)
            first = data.chunks[0]
            if sample <= int(first.shape[0]) and not data.vi.row_prefix_enabled:
                mean_dp = np.asarray(first[:sample], dtype=np.float32).mean(axis=0)
            else:
                # Use a contiguous prefix so cold startup touches only the first
                # chunks. Evenly spaced samples estimate the same center, but page
                # in the whole 19 GB stack and cost as much as exact detection.
                indices = np.arange(sample, dtype=np.uint32)
                mean_dp = data.vi.mean_frames(indices)
        else:
            mean_dp = data.vi.detector_sum()
        mean_dp = np.asarray(mean_dp, dtype=np.float32)
        threshold = float(mean_dp.mean()) + float(mean_dp.std())
        mask = mean_dp > threshold
        total = int(mask.sum())
        if total == 0:
            return self
        rows = np.arange(mean_dp.shape[0], dtype=np.float32)[:, None]
        cols = np.arange(mean_dp.shape[1], dtype=np.float32)[None, :]
        cx = float((cols * mask).sum() / total)
        cy = float((rows * mask).sum() / total)
        radius = float(round(np.sqrt(total / np.pi)))
        self.center_col, self.center_row, self.bf_radius = cx, cy, radius
        if update_roi:
            self.roi_center_col = cx
            self.roi_center_row = cy
            if self.fast_interaction and self.fast_interaction_ready:
                self._clear_virtual_image_caches()
                self._compute_virtual_image_from_roi()
        return self

    def _wire_multi_dataset(self):
        # Lazy-loaded 5D stack: start the frame slider spanning only the decoded
        # datasets (1 at first) so the user can NEVER slide onto a not-yet-decoded
        # slot. As each background decode lands, on_ready grows n_frames and
        # refreshes the loading banner in the title.
        data = self._data
        if not hasattr(data, "on_ready") or not hasattr(data, "n_ready"):
            self._multi = None
            return
        self._multi = data
        n_total = len(data.datasets)
        self._multi_total = n_total
        self.frame_dim_label = "Dataset"
        # slider only spans what's decoded; frame_idx 0 is dataset 0
        self.n_frames = max(1, data.n_ready)
        self._refresh_multi_title()
        # capture the kernel IOLoop so the background decode thread can push trait
        # updates safely (traits must be set on the loop that owns the comm).
        try:
            from tornado.ioloop import IOLoop
            self._ioloop = IOLoop.current()
        except Exception:
            self._ioloop = None
        data.on_ready = self._on_multi_dataset_ready

    def _refresh_multi_title(self):
        # Title = the CURRENT dataset's file name (so the operator always knows
        # which file they're looking at as they flip). While the background is
        # still decoding, append a "loading k/N" tail; once every dataset is in,
        # the tail disappears and the title is just the file name.
        m = self._multi
        name = m.names[m.active_idx] if m.names else f"dataset {m.active_idx}"
        n_ready, n_total = m.n_ready, self._multi_total
        self.title = name if n_ready >= n_total else f"{name}  -  loading {n_ready}/{n_total}"

    def _on_multi_dataset_ready(self, idx: int):
        # Called from the background decode thread. Hop to the kernel IOLoop so the
        # n_frames + title trait writes sync cleanly to the frontend.
        def _apply():
            self.n_frames = max(1, self._multi.n_ready)
            self._refresh_multi_title()
        if self._ioloop is not None:
            self._ioloop.add_callback(_apply)
        else:
            _apply()

    def _on_frame_idx_change(self, change=None):
        # 5D multi-dataset: point the proxy at the slid-to dataset BEFORE the base
        # recompute reads self._data.vi / .frame(). If that dataset isn't decoded
        # yet, set_active holds the last ready one. Refresh the title so it names
        # the dataset actually on screen.
        data = self._data
        if hasattr(data, "set_active"):
            data.set_active(int(self.frame_idx))
            if getattr(self, "_multi", None) is not None:
                self._refresh_multi_title()
        return super()._on_frame_idx_change(change)

    def _get_frame(self, row: int, col: int) -> np.ndarray:
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return super()._get_frame(row, col)
        return data.frame(row * self.shape_cols + col)

    def _fast_masked_sum(self, mask):
        import torch
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return super()._fast_masked_sum(mask)
        mask_np = mask.detach().cpu().numpy() if hasattr(mask, "detach") else np.asarray(mask)
        if self.fast_interaction and self.fast_interaction_ready and data.fast_vi is not None:
            self._ensure_fast_interaction_ready()
            vi = data.fast_vi.masked_sum(_bin_mask(mask_np, data.fast_bin))
        else:
            vi = data.vi.masked_sum(mask_np)  # (N,) int32, raw Metal
        return torch.from_numpy(vi.astype(np.float32, copy=False)).reshape(self._scan_shape)

    def _detector_mask_np(self) -> np.ndarray | None:
        cx = float(self.roi_center_col)
        cy = float(self.roi_center_row)
        rows = self._det_row_coords_np
        cols = self._det_col_coords_np

        if self.roi_mode == "point":
            # single detector pixel under the marker - the one pixel whose cell
            # contains (cx, cy). Without this the virtual image is empty in point mode.
            return (np.abs(cols - cx) < 0.5) & (np.abs(rows - cy) < 0.5)
        if self.roi_mode == "circle" and self.roi_radius > 0:
            radius = float(self.roi_radius)
            return (cols - cx) ** 2 + (rows - cy) ** 2 <= radius ** 2
        if self.roi_mode == "square" and self.roi_radius > 0:
            half_size = float(self.roi_radius)
            return (np.abs(cols - cx) <= half_size) & (np.abs(rows - cy) <= half_size)
        if self.roi_mode == "annular" and self.roi_radius > 0:
            inner = float(self.roi_radius_inner)
            outer = float(self.roi_radius)
            dist_sq = (cols - cx) ** 2 + (rows - cy) ** 2
            return (dist_sq >= inner ** 2) & (dist_sq <= outer ** 2)
        if self.roi_mode == "rect" and self.roi_width > 0 and self.roi_height > 0:
            half_width = float(self.roi_width) / 2.0
            half_height = float(self.roi_height) / 2.0
            return (
                (np.abs(cols - cx) <= half_width)
                & (np.abs(rows - cy) <= half_height)
            )
        return None

    def _set_virtual_image_bytes_np(self, vi: np.ndarray):
        arr = np.asarray(vi).reshape(self._scan_shape)
        arr = np.asarray(arr, dtype=np.float32, order="C")
        self.virtual_image_bytes = arr.tobytes()

    def _set_virtual_image_startup_preview(self):
        """Show a cheap non-black placeholder while fast BF is building."""
        rows = np.linspace(0.0, 1.0, self.shape_rows, dtype=np.float32)[:, None]
        cols = np.linspace(0.0, 1.0, self.shape_cols, dtype=np.float32)[None, :]
        # Use the current DP mean as the physical scale, with a tiny deterministic
        # gradient so min/max normalization does not collapse to black.
        try:
            frame = self._get_frame(self.shape_rows // 2, self.shape_cols // 2)
            level = float(np.asarray(frame, dtype=np.float32).mean())
        except Exception:
            level = 1.0
        scale = max(level, 1.0)
        preview = scale * (0.95 + 0.05 * (rows + cols))
        self._set_virtual_image_bytes_np(preview)

    def set_fast_interaction(self, enabled: bool = True, *, wait: bool = True):
        """Toggle bin2 fast interaction for BF/DF/ADF virtual images.

        The first enable builds a detector-bin2 sidecar. Cursor diffraction
        patterns continue to come from the full 192x192 raw chunks.
        """
        enabled = bool(enabled)
        if enabled and wait:
            self._ensure_fast_interaction_ready()
        self._suppress_fast_interaction_observer = True
        try:
            self.fast_interaction = enabled
        finally:
            self._suppress_fast_interaction_observer = False
        self._clear_virtual_image_caches()
        self._compute_virtual_image_from_roi()
        if enabled and not wait:
            self._start_fast_interaction_background()
        return self

    def _on_fast_interaction_change(self, change=None):
        if getattr(self, "_suppress_fast_interaction_observer", False):
            return
        if self.fast_interaction and self._fast_interaction_async:
            self._start_fast_interaction_background()
        elif self.fast_interaction:
            self._ensure_fast_interaction_ready()
        self._clear_virtual_image_caches()
        self._compute_virtual_image_from_roi()

    def _on_roi_change(self, change=None):
        if getattr(self, "_mps_initializing", False):
            return
        return super()._on_roi_change(change)

    def _on_roi_center_change(self, change=None):
        if getattr(self, "_mps_initializing", False):
            return
        return super()._on_roi_center_change(change)

    def _clear_virtual_image_caches(self):
        self._cached_bf_virtual = None
        self._cached_abf_virtual = None
        self._cached_adf_virtual = None
        self._cached_haadf_virtual = None

    def _get_cached_preset(self):
        if abs(self.roi_center_col - self.center_col) >= 1:
            return None
        if abs(self.roi_center_row - self.center_row) >= 1:
            return None

        bf = float(self.bf_radius)
        if self.roi_mode == "circle" and abs(self.roi_radius - bf) < 1:
            return self._cached_bf_virtual
        if (
            self.roi_mode == "annular"
            and abs(self.roi_radius_inner - bf * 0.5) < 1
            and abs(self.roi_radius - bf) < 1
        ):
            return self._cached_abf_virtual
        if (
            self.roi_mode == "annular"
            and abs(self.roi_radius_inner - bf) < 1
            and abs(self.roi_radius - bf * 2.0) < 1
        ):
            return self._cached_adf_virtual
        if (
            self.roi_mode == "annular"
            and abs(self.roi_radius_inner - bf * 2.0) < 1
            and abs(self.roi_radius - bf * 4.0) < 1
        ):
            return self._cached_haadf_virtual
        return None

    def _preset_mask_np(self, name: str) -> np.ndarray | None:
        rows = self._det_row_coords_np
        cols = self._det_col_coords_np
        cx = float(self.center_col)
        cy = float(self.center_row)
        bf = float(max(1.0, self.bf_radius))
        dist_sq = (cols - cx) ** 2 + (rows - cy) ** 2
        preset = str(name).strip().lower()
        if preset == "bf":
            return dist_sq <= bf ** 2
        if preset == "abf":
            return (dist_sq >= (bf * 0.5) ** 2) & (dist_sq <= bf ** 2)
        if preset == "adf":
            return (dist_sq >= bf ** 2) & (dist_sq <= (bf * 2.0) ** 2)
        if preset == "haadf":
            return (dist_sq >= (bf * 2.0) ** 2) & (dist_sq <= (bf * 4.0) ** 2)
        return None

    def _cache_fast_presets(self):
        data = self._data
        if (
            not isinstance(data, ChunkedFrames)
            or data.fast_vi is None
            or not self.fast_interaction_ready
        ):
            return

        for name, attr in (
            ("bf", "_cached_bf_virtual"),
            ("abf", "_cached_abf_virtual"),
            ("adf", "_cached_adf_virtual"),
            ("haadf", "_cached_haadf_virtual"),
        ):
            mask = self._preset_mask_np(name)
            if mask is None:
                continue
            vi = data.fast_vi.masked_sum(_bin_mask(mask, data.fast_bin))
            arr = np.asarray(vi).reshape(self._scan_shape)
            arr = np.asarray(arr, dtype=np.float32, order="C")
            setattr(self, attr, arr.tobytes())

    def _ensure_fast_interaction_ready(self) -> bool:
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return False
        if int(getattr(data, "det_bin", 1)) > 1:
            self.fast_interaction_ready = True
            return True
        data.ensure_fast_interaction(verbose=self._fast_interaction_verbose)
        self.fast_interaction_ready = True
        return True

    def _start_fast_interaction_background(self):
        if self.fast_interaction_ready or self.fast_interaction_building:
            return
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return
        self.fast_interaction_building = True
        self._fast_interaction_error = None

        def _build():
            if self._fast_interaction_async:
                time.sleep(0.05)
            try:
                data.ensure_fast_interaction(verbose=self._fast_interaction_verbose)
                self.fast_interaction_ready = True
                self._clear_virtual_image_caches()
                self._cache_fast_presets()
                if self.fast_interaction:
                    self._compute_virtual_image_from_roi()
                    if getattr(self, "vi_roi_mode", "off") != "off":
                        self._compute_vi_roi_dp()
            except Exception as exc:  # pragma: no cover - surfaced in notebooks
                self._fast_interaction_error = repr(exc)
            finally:
                self.fast_interaction_building = False

        self._fast_interaction_thread = threading.Thread(
            target=_build,
            name="Show4DSTEMMPS-fast-interaction",
            daemon=True,
        )
        self._fast_interaction_thread.start()

    def wait_for_fast_interaction(self, timeout: float | None = None) -> bool:
        thread = self._fast_interaction_thread
        if thread is not None:
            thread.join(timeout)
        if self._fast_interaction_error is not None:
            raise RuntimeError(self._fast_interaction_error)
        return bool(self.fast_interaction_ready)

    def _start_radial_interaction_background(self):
        data = self._data
        if self.fast_interaction or not isinstance(data, ChunkedFrames):
            return
        if not data.vi.row_prefix_enabled:
            return
        center_row = float(self.roi_center_row)
        center_col = float(self.roi_center_col)
        if data.vi.radial_cache_ready(center_row, center_col):
            self.radial_interaction_ready = True
            self._radial_interaction_pending_center = None
            return
        self.radial_interaction_ready = False
        self._radial_interaction_request += 1
        self._radial_interaction_pending_center = (center_row, center_col)
        if self.radial_interaction_building:
            return
        self.radial_interaction_building = True
        self._radial_interaction_error = None

        def _build():
            try:
                while True:
                    request = self._radial_interaction_request
                    center = self._radial_interaction_pending_center
                    if center is None:
                        return
                    time.sleep(_RADIAL_INTERACTION_IDLE_DELAY)
                    if (
                        request != self._radial_interaction_request
                        or center != self._radial_interaction_pending_center
                    ):
                        continue
                    data.vi._ensure_radial_cache(center[0], center[1])
                    if (
                        request == self._radial_interaction_request
                        and center == self._radial_interaction_pending_center
                    ):
                        self.radial_interaction_ready = True
                        self._radial_interaction_pending_center = None
                        return
            except Exception as exc:  # pragma: no cover - surfaced in notebooks
                self._radial_interaction_error = repr(exc)
            finally:
                self.radial_interaction_building = False

        self._radial_interaction_thread = threading.Thread(
            target=_build,
            name="Show4DSTEMMPS-radial-interaction",
            daemon=True,
        )
        self._radial_interaction_thread.start()

    def wait_for_radial_interaction(self, timeout: float | None = None) -> bool:
        thread = self._radial_interaction_thread
        if thread is not None:
            thread.join(timeout)
        if self._radial_interaction_error is not None:
            raise RuntimeError(self._radial_interaction_error)
        return bool(self.radial_interaction_ready)

    def _compute_virtual_image_from_roi(self):
        data = self._data
        if getattr(self, "_mps_initializing", False):
            self.virtual_image_bytes = b""
            return
        if not isinstance(data, ChunkedFrames):
            return super()._compute_virtual_image_from_roi()
        cached = self._get_cached_preset()
        if cached is not None:
            self.virtual_image_bytes = cached
            return
        if (
            self.fast_interaction
            and self._fast_interaction_async
            and not self.fast_interaction_ready
        ):
            self._set_virtual_image_startup_preview()
            return
        if (
            self.fast_interaction
            and not self.fast_interaction_ready
            and not self._fast_interaction_async
        ):
            self._ensure_fast_interaction_ready()

        start_radial_background = False
        if (
            not self.fast_interaction
            and self.roi_mode in ("circle", "annular")
            and float(self.roi_radius) > 0
        ):
            inner = float(self.roi_radius_inner) if self.roi_mode == "annular" else 0.0
            radial_vi = data.vi.radial_masked_sum(
                center_row=float(self.roi_center_row),
                center_col=float(self.roi_center_col),
                outer_radius=float(self.roi_radius),
                inner_radius=inner,
                build=False,
            )
            if radial_vi is not None:
                self._set_virtual_image_bytes_np(radial_vi)
                return
            start_radial_background = True

        mask = self._detector_mask_np()
        if mask is None:
            row = int(max(0, min(round(float(self.roi_center_row)), self.det_rows - 1)))
            col = int(max(0, min(round(float(self.roi_center_col)), self.det_cols - 1)))
            if self.fast_interaction and self.fast_interaction_ready and data.fast_vi is not None:
                self._ensure_fast_interaction_ready()
                fast_rows, fast_cols = data.fast_vi.det
                scale_r = self.det_rows / fast_rows
                scale_c = self.det_cols / fast_cols
                fast_row = int(max(0, min(round(row / scale_r), fast_rows - 1)))
                fast_col = int(max(0, min(round(col / scale_c), fast_cols - 1)))
                fast_mask = np.zeros((fast_rows, fast_cols), dtype=bool)
                fast_mask[fast_row, fast_col] = True
                self._set_virtual_image_bytes_np(data.fast_vi.masked_sum(fast_mask))
            else:
                self._set_virtual_image_bytes_np(data.column(row, col))
            if start_radial_background:
                self._start_radial_interaction_background()
            return

        if self.fast_interaction and self.fast_interaction_ready and data.fast_vi is not None:
            self._ensure_fast_interaction_ready()
            vi = data.fast_vi.masked_sum(_bin_mask(mask, data.fast_bin))
        else:
            vi = data.vi.masked_sum(mask)
        self._set_virtual_image_bytes_np(vi)
        if start_radial_background:
            self._start_radial_interaction_background()

    def _clear_vi_roi_dp(self):
        if hasattr(self, "vi_roi_dp_bytes"):
            self.vi_roi_dp_bytes = b""
        if hasattr(self, "summed_dp_bytes"):
            self.summed_dp_bytes = b""
        if hasattr(self, "summed_dp_count"):
            self.summed_dp_count = 0

    def _set_vi_roi_dp(self, dp: np.ndarray, n_positions: int):
        payload = np.asarray(dp, dtype=np.float32, order="C").tobytes()
        if hasattr(self, "vi_roi_dp_bytes"):
            self.vi_roi_dp_bytes = payload
        if hasattr(self, "summed_dp_bytes"):
            self.summed_dp_bytes = payload
        if hasattr(self, "summed_dp_count"):
            self.summed_dp_count = int(n_positions)

    def _vi_roi_indices_np(self) -> np.ndarray:
        rows = np.arange(self.shape_rows, dtype=np.float32)[:, None]
        cols = np.arange(self.shape_cols, dtype=np.float32)[None, :]
        center_row = float(self.vi_roi_center_row)
        center_col = float(self.vi_roi_center_col)

        if self.vi_roi_mode == "point":
            # single scan position under the marker -> summed DP is that one frame
            mask = (np.abs(rows - center_row) < 0.5) & (np.abs(cols - center_col) < 0.5)
        elif self.vi_roi_mode == "circle":
            radius = float(self.vi_roi_radius)
            mask = (rows - center_row) ** 2 + (cols - center_col) ** 2 <= radius ** 2
        elif self.vi_roi_mode == "square":
            half_size = float(self.vi_roi_radius)
            mask = (
                (np.abs(rows - center_row) <= half_size)
                & (np.abs(cols - center_col) <= half_size)
            )
        elif self.vi_roi_mode == "rect":
            half_w = float(self.vi_roi_width) / 2.0
            half_h = float(self.vi_roi_height) / 2.0
            mask = (
                (np.abs(rows - center_row) <= half_h)
                & (np.abs(cols - center_col) <= half_w)
            )
        else:
            return np.empty(0, dtype=np.uint32)
        return np.flatnonzero(mask.reshape(-1)).astype(np.uint32, copy=False)

    def _compute_summed_dp_from_vi_roi(self):
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return super()._compute_summed_dp_from_vi_roi()
        if self.vi_roi_mode == "off":
            self._clear_vi_roi_dp()
            return

        indices = self._vi_roi_indices_np()
        n_positions = int(indices.size)
        if n_positions == 0:
            self._clear_vi_roi_dp()
            return

        if self.fast_interaction and self.fast_interaction_ready and data.fast_vi is not None:
            self._ensure_fast_interaction_ready()
            dp = data.fast_vi.mean_frames(indices)
            dp = _upsample_bin_dp(dp, (self.det_rows, self.det_cols), data.fast_bin)
        else:
            dp = data.vi.mean_frames(indices)

        self._set_vi_roi_dp(dp, n_positions)

    def _compute_vi_roi_dp(self):
        data = self._data
        if not isinstance(data, ChunkedFrames):
            return super()._compute_vi_roi_dp()
        if self.vi_roi_mode == "off":
            self._clear_vi_roi_dp()
            return

        indices = self._vi_roi_indices_np()
        n_positions = int(indices.size)
        if n_positions == 0:
            self._clear_vi_roi_dp()
            return

        reduce = getattr(self, "vi_roi_reduce", "mean")
        if reduce in ("mean", "sum"):
            if self.fast_interaction and self.fast_interaction_ready and data.fast_vi is not None:
                self._ensure_fast_interaction_ready()
                dp = data.fast_vi.mean_frames(indices)
                if reduce == "sum":
                    dp = dp * float(n_positions)
                dp = _upsample_bin_dp(dp, (self.det_rows, self.det_cols), data.fast_bin)
            else:
                dp = data.vi.mean_frames(indices)
                if reduce == "sum":
                    dp = dp * float(n_positions)
        elif reduce == "max":
            dp = np.full((self.det_rows, self.det_cols), -np.inf, dtype=np.float32)
            for idx in indices:
                np.maximum(dp, data.frame(int(idx)), out=dp)
        else:
            return

        self._set_vi_roi_dp(dp, n_positions)


def load_4dstem_mps(
    master_path: str,
    *,
    scan_shape=None,
    fast_interaction: bool = True,
    fast_interaction_async: bool = True,
    full_resolution_interaction: bool = False,
    det_bin: int = 1,
    compact: bool = False,
    compact_target_gb: float = _DEFAULT_COMPACT_TARGET_BYTES / 1e9,
    auto_detect_frames: int | None = 64,
    initial_preset: str | None = "BF",
    **kwargs,
):
    """Load full no-bin DP display plus the supported fast VI path."""
    verbose = kwargs.pop("verbose", True)
    if full_resolution_interaction:
        raise ValueError(
            "full_resolution_interaction has been disabled. Use load(...) and "
            "show_4dstem_mps(...) for fast interaction."
        )
    from quantem.widget.io import load_mps_4dstem

    data = load_mps_4dstem(
        master_path,
        scan_shape=scan_shape,
        verbose=verbose,
        compact=compact,
        compact_target_gb=compact_target_gb,
        det_bin=det_bin,
    )
    return show_4dstem_mps(
        data,
        scan_shape=scan_shape,
        fast_interaction=fast_interaction,
        fast_interaction_async=fast_interaction_async,
        full_resolution_interaction=full_resolution_interaction,
        fast_interaction_verbose=verbose,
        auto_detect_frames=auto_detect_frames,
        initial_preset=initial_preset,
        verbose=verbose,
        **kwargs,
    )


def show_4dstem_mps(
    data,
    *,
    scan_shape=None,
    fast_interaction: bool = True,
    fast_interaction_async: bool = True,
    full_resolution_interaction: bool = False,
    fast_interaction_verbose: bool = True,
    auto_detect_frames: int | None = 64,
    initial_preset: str | None = "BF",
    verbose: bool = True,
    **kwargs,
):
    """Wrap preloaded MPS chunks in the no-copy raw-Metal viewer."""
    if full_resolution_interaction:
        raise ValueError(
            "full_resolution_interaction has been disabled. Use load(...) and "
            "show_4dstem_mps(...) for fast interaction."
        )
    if isinstance(data, ChunkedFrames):
        frames = data
    else:
        if hasattr(data, "scan_shape") and scan_shape is None:
            scan_shape = data.scan_shape
        row_prefix = bool(
            getattr(data, "row_prefix", False)
            or getattr(data, "metadata", {}).get("row_prefix", False)
        )
        frames = ChunkedFrames(data, row_prefix=row_prefix)
    if scan_shape is None and hasattr(data, "metadata"):
        scan_shape = data.metadata.get("scan_shape")
    return Show4DSTEMMPS(
        frames,
        scan_shape=scan_shape,
        fast_interaction=fast_interaction,
        fast_interaction_async=fast_interaction_async,
        full_resolution_interaction=full_resolution_interaction,
        fast_interaction_verbose=fast_interaction_verbose,
        auto_detect_frames=auto_detect_frames,
        initial_preset=initial_preset,
        verbose=verbose,
        **kwargs,
    )


def _meta_number(meta: dict, *keys: str):
    for key in keys:
        if key not in meta:
            continue
        value = meta[key]
        try:
            arr = np.asarray(value)
            if arr.size == 1:
                return float(arr.reshape(-1)[0])
        except Exception:
            try:
                return float(value)
            except Exception:
                pass
    return None


def Show4DSTEM_MACBOOK(
    data,
    meta: dict | None = None,
    *,
    scan_sampling_A: float | None = None,
    det_sampling_mrad_per_px: float | None = None,
    semiangle_mrad: float | None = None,
    sampling=None,
    units=None,
    **kwargs,
):
    """Local MacBook raw-Metal 4D-STEM viewer.

    This is a deliberately explicit alias for ``show_4dstem_mps``: it keeps the
    full 192x192 diffraction patterns local in unified Metal memory and uses
    the fused detector-bin2 sidecar for real-time BF/DF/ADF interaction.

    Pass ``scan_sampling_A`` and either ``det_sampling_mrad_per_px`` or
    ``semiangle_mrad`` to show scan coordinates in Angstrom and detector
    coordinates in mrad. If ``semiangle_mrad`` is provided, the detector
    sampling is inferred after BF-radius detection as
    ``semiangle_mrad / bf_radius_px``.
    """
    combined_meta = {}
    if hasattr(data, "metadata"):
        combined_meta.update(getattr(data, "metadata", {}) or {})
    if meta:
        combined_meta.update(meta)
    if scan_sampling_A is None:
        scan_sampling_A = _meta_number(
            combined_meta,
            "scan_sampling_A",
            "scan_sampling",
            "pixel_size_A",
        )
    if det_sampling_mrad_per_px is None:
        det_sampling_mrad_per_px = _meta_number(
            combined_meta,
            "det_sampling_mrad_per_px",
            "detector_sampling_mrad_per_px",
            "k_pixel_size",
        )
    if semiangle_mrad is None:
        semiangle_mrad = _meta_number(combined_meta, "semiangle_mrad", "semiangle")
    if sampling is None and (scan_sampling_A is not None or det_sampling_mrad_per_px is not None):
        sampling = (
            float(scan_sampling_A) if scan_sampling_A is not None else 1.0,
            float(det_sampling_mrad_per_px) if det_sampling_mrad_per_px is not None else 1.0,
        )
    if units is None and sampling is not None:
        units = (
            "Å" if scan_sampling_A is not None else "pixels",
            "mrad" if det_sampling_mrad_per_px is not None else "pixels",
        )
    verbose = bool(kwargs.get("verbose", True))
    viewer = show_4dstem_mps(data, sampling=sampling, units=units, **kwargs)
    inferred_det_sampling = False
    if det_sampling_mrad_per_px is None and semiangle_mrad is not None:
        bf_radius = float(getattr(viewer, "bf_radius", 0) or 0)
        if bf_radius > 0:
            det_sampling_mrad_per_px = float(semiangle_mrad) / bf_radius
            viewer.k_pixel_size = float(det_sampling_mrad_per_px)
            viewer.k_pixel_unit = "mrad"
            inferred_det_sampling = True
    if scan_sampling_A is not None:
        viewer.pixel_size = float(scan_sampling_A)
        viewer.pixel_unit = "Å"
    viewer.macbook_sampling = {
        "scan_sampling_A": scan_sampling_A,
        "det_sampling_mrad_per_px": det_sampling_mrad_per_px,
        "semiangle_mrad": semiangle_mrad,
        "det_sampling_inferred_from_bf_radius": inferred_det_sampling,
    }
    if verbose and (scan_sampling_A is not None or det_sampling_mrad_per_px is not None):
        parts = []
        if scan_sampling_A is not None:
            parts.append(f"scan {float(scan_sampling_A):.4g} Å/px")
        if det_sampling_mrad_per_px is not None:
            source = " from BF radius" if inferred_det_sampling else ""
            parts.append(f"detector {float(det_sampling_mrad_per_px):.4g} mrad/px{source}")
        print(f"MacBook sampling: {', '.join(parts)}")
    return viewer


def _drop_cached_decompressor():
    """Release decoder scratch buffers before allocating interaction sidecars."""
    from quantem.widget.io import clear_mps_cache

    clear_mps_cache()
