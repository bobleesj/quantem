"""
Show3DVolume: orthogonal slice viewer for 3D volumetric data.

Displays XY, XZ, YZ planes with interactive sliders. All slicing happens
in JavaScript for instant response. Useful for ptychography reconstruction
volumes, tomograms, and any voxel data where the user wants three-plane
inspection rather than a frame stack.
"""
import json
import math
import pathlib
from numbers import Real
from typing import Self

import anywidget
import numpy as np
import traitlets

from quantem.widget.array_utils import to_numpy
from quantem.widget.show2d import _reject_unknown_kwargs
from quantem.widget.state import (
    build_json_header,
    resolve_widget_version,
    save_state_file,
    unwrap_state_payload,
)


# Names that JS bundle's GPUColormapEngine knows. Keep in sync with
# js/colormaps.ts COLORMAPS table.
_VALID_CMAPS = frozenset({
    "inferno", "viridis", "plasma", "magma", "hot", "gray", "hsv", "turbo",
    "cividis", "RdBu", "RdBu_r", "seismic", "twilight", "twilight_shifted",
})


class Show3DVolume(anywidget.AnyWidget):
    """3D volume viewer with three orthogonal slice planes.

    Parameters
    ----------
    data : array_like
        3D array of shape (nz, ny, nx).
    data_b : array_like, optional
        Second volume for side-by-side comparison. Must match `data` shape.
    title : str, optional
        Title displayed above the viewer.
    title_b : str, optional
        Title for volume B (dual mode).
    cmap : str, default "inferno"
        Colormap name. One of {valid set above}.
    pixel_size : float or sequence of 3 floats, optional
        Voxel sampling in angstroms. Pass a scalar for isotropic data, or a
        3-tuple `(pz, py, px)` for anisotropic data (e.g. multislice ptycho
        with z-thickness >> xy-sampling). Per-axis values flow to JS via the
        `pixel_size_axes` trait for correct scale bars on each panel.
    show_stats : bool, default True
        Compute per-slice statistics traits on each slice change (`widget.stats_mean`,
        `stats_min`, `stats_max`, `stats_std`, each a list of 3 floats: XY/XZ/YZ).
        Python-side only; the JS widget does not render a stats bar. Set False to
        skip 12 reductions per slice scrub on multi-MB volumes when you don't
        need the values.
    show_controls : bool, default True
        Show the secondary control row (Z stretch, Color, Smooth, Colorbar).
        Top toolbar (FFT, Export, Copy, Reset) and the slice/playback sliders are
        always visible.
    show_crosshair : bool, default False
        Deprecated compatibility no-op. Crosshair overlays are no longer rendered.
    show_fft : bool, default False
        Toggle FFT panel for the active plane.
    show_diff : bool, default False
        In dual mode, display |A - B| (absolute difference) as a third row.
    log_scale : bool, default False
        Use signed log1p for intensity mapping.
    auto_contrast : bool, default False
        Use percentile-based contrast (2nd-98th).
    vmin, vmax : float, optional
        Manual contrast limits.
    fps : float, default 5.0
        Playback speed when scrubbing one axis.
    play_axis : int, default 0
        Which axis to animate (0=Z, 1=Y, 2=X, 3=cycle all).
    dim_labels : list of str, optional
        Labels for data axes 0, 1, 2 in that order. Default ["Z", "Y", "X"]
        matches numpy-style indexing (axis 0 is the slice dim). For multislice
        ptycho with shape (nz, ny, nx), pass ["Z (slice)", "Y", "X"].

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show3DVolume
    >>> volume = np.random.rand(64, 64, 64).astype(np.float32)
    >>> Show3DVolume(volume, title="My Volume", cmap="viridis")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show3dvolume.js"
    _widget_name = "Show3DVolume"
    _viewer_kind = "volume"

    widget_version = traitlets.Unicode("unknown").tag(sync=True)
    viewer_kind = traitlets.Unicode("volume").tag(sync=True)

    # Volume dimensions
    nx = traitlets.Int(1).tag(sync=True)
    ny = traitlets.Int(1).tag(sync=True)
    nz = traitlets.Int(1).tag(sync=True)
    # Slice positions
    slice_x = traitlets.CInt(0).tag(sync=True)
    slice_y = traitlets.CInt(0).tag(sync=True)
    slice_z = traitlets.CInt(0).tag(sync=True)
    # Raw volume data (sent once)
    volume_bytes = traitlets.Bytes(b"").tag(sync=True)
    # Dual-volume comparison mode
    volume_bytes_b = traitlets.Bytes(b"").tag(sync=True)
    title_b = traitlets.Unicode("").tag(sync=True)
    dual_mode = traitlets.Bool(False).tag(sync=True)
    show_diff = traitlets.Bool(False).tag(sync=True)
    linked_contrast = traitlets.Bool(True).tag(sync=True)
    # Stats for volume B (3 values: xy, xz, yz)
    # stats_*_b: programmatic Python access only (no JS consumer). Don't sync,
    # otherwise every slice scrub ships 4 lists of 3 floats across the websocket
    # for nothing.
    stats_mean_b = traitlets.List(traitlets.Float())
    stats_min_b = traitlets.List(traitlets.Float())
    stats_max_b = traitlets.List(traitlets.Float())
    stats_std_b = traitlets.List(traitlets.Float())
    # Display
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    # Scale bar. `pixel_size` is a scalar (lateral sampling, used by XY/XZ/YZ width-axis
    # scale bars). `pixel_size_axes` is the full per-axis triple [z, y, x] in the same
    # units — populated from tuple/list input; defaults to [pixel_size]*3 for scalar input.
    # Both sync to JS; JS uses pixel_size_axes when present for depth-axis scale bars
    # and falls back to pixel_size for the lateral axes.
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    pixel_size_axes = traitlets.List(traitlets.Float(), default_value=[0.0, 0.0, 0.0]).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    # Depth-axis display stretch for non-cubic volumes (CSS-only, zero memory).
    # Scales XZ/YZ panel display height. Useful when nz << nxy (e.g. multislice
    # ptycho with nz=14, nxy=730 → set z_stretch high to make depth panels readable).
    z_stretch = traitlets.Float(1.0).tag(sync=True)
    # UI
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    # Deprecated compatibility no-op. Crosshair overlays are no longer rendered.
    show_crosshair = traitlets.Bool(False).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    orthographic = traitlets.Bool(False).tag(sync=True)
    smooth = traitlets.Bool(False).tag(sync=True)
    # Deprecated compatibility no-op. The JS widget always renders the compact layout.
    compact = traitlets.Bool(True).tag(sync=True)
    flip = traitlets.Bool(False).tag(sync=True)
    # Axis labels (dim 0, 1, 2)
    # Default labels follow numpy-style data-axis order: axis 0 is the slice dim (Z),
    # axis 1 is Y, axis 2 is X. The XY/XZ/YZ panel headers display
    # "<dl[1]><dl[2]> (<dl[0]>=...)" etc, so default labels show as e.g. "YX (Z=7)".
    dim_labels = traitlets.List(traitlets.Unicode(), default_value=["Z", "Y", "X"]).tag(sync=True)
    # Stats (3 values: xy, xz, yz)
    # stats_*: programmatic Python access only (no JS consumer). Don't sync.
    stats_mean = traitlets.List(traitlets.Float())
    stats_min = traitlets.List(traitlets.Float())
    stats_max = traitlets.List(traitlets.Float())
    stats_std = traitlets.List(traitlets.Float())
    # Playback
    playing = traitlets.Bool(False).tag(sync=True)
    reverse = traitlets.Bool(False).tag(sync=True)
    boomerang = traitlets.Bool(False).tag(sync=True)
    fps = traitlets.Float(5.0).tag(sync=True)
    loop = traitlets.Bool(True).tag(sync=True)
    play_axis = traitlets.Int(0).tag(sync=True)  # 0=Z, 1=Y, 2=X, 3=All
    # Export
    _export_axis = traitlets.Int(0).tag(sync=True)
    _gif_export_requested = traitlets.Bool(False).tag(sync=True)
    _gif_data = traitlets.Bytes(b"").tag(sync=True)
    _gif_metadata_json = traitlets.Unicode("").tag(sync=True)
    _zip_export_requested = traitlets.Bool(False).tag(sync=True)
    _zip_data = traitlets.Bytes(b"").tag(sync=True)

    # Validators (consistent with Show3D)

    @traitlets.validate("cmap")
    def _validate_cmap(self, proposal):
        val = str(proposal["value"])
        if val not in _VALID_CMAPS:
            raise traitlets.TraitError(
                f"Unknown cmap {val!r}. Valid: {sorted(_VALID_CMAPS)}"
            )
        return val

    @traitlets.validate("fps")
    def _validate_fps(self, proposal):
        val = float(proposal["value"])
        if not math.isfinite(val):
            raise traitlets.TraitError(f"fps must be finite, got {val}")
        if val <= 0:
            raise traitlets.TraitError(f"fps must be > 0, got {val}")
        return val

    @traitlets.validate("pixel_size")
    def _validate_pixel_size(self, proposal):
        val = float(proposal["value"])
        if math.isnan(val) or math.isinf(val):
            raise traitlets.TraitError(f"pixel_size must be finite, got {val}")
        if val < 0:
            raise traitlets.TraitError(f"pixel_size must be >= 0, got {val}")
        return val

    @traitlets.validate("pixel_size_axes")
    def _validate_pixel_size_axes(self, proposal):
        val = [float(v) for v in proposal["value"]]
        if len(val) != 3:
            raise traitlets.TraitError(
                f"pixel_size_axes must have length 3, got {len(val)}"
            )
        for v in val:
            if not math.isfinite(v):
                raise traitlets.TraitError(f"pixel_size_axes values must be finite, got {val}")
            if v < 0:
                raise traitlets.TraitError(f"pixel_size_axes values must be >= 0, got {val}")
        return val

    @traitlets.validate("play_axis")
    def _validate_play_axis(self, proposal):
        val = int(proposal["value"])
        if val not in (0, 1, 2, 3):
            raise traitlets.TraitError(f"play_axis must be 0/1/2/3, got {val}")
        return val

    @traitlets.validate("_export_axis")
    def _validate_export_axis(self, proposal):
        val = int(proposal["value"])
        if val not in (0, 1, 2):
            raise traitlets.TraitError(f"_export_axis must be 0/1/2, got {val}")
        return val

    @traitlets.validate("z_stretch")
    def _validate_z_stretch(self, proposal):
        val = float(proposal["value"])
        if math.isnan(val) or math.isinf(val):
            raise traitlets.TraitError(f"z_stretch must be finite, got {val}")
        return max(1.0, min(val, 30.0))

    @traitlets.validate("dim_labels")
    def _validate_dim_labels(self, proposal):
        val = list(proposal["value"])
        if len(val) != 3:
            raise traitlets.TraitError(
                f"dim_labels must have length 3, got {len(val)}"
            )
        return val

    @traitlets.validate("slice_z")
    def _validate_slice_z(self, proposal):
        return max(0, min(int(proposal["value"]), max(0, int(self.nz) - 1)))

    @traitlets.validate("slice_y")
    def _validate_slice_y(self, proposal):
        return max(0, min(int(proposal["value"]), max(0, int(self.ny) - 1)))

    @traitlets.validate("slice_x")
    def _validate_slice_x(self, proposal):
        return max(0, min(int(proposal["value"]), max(0, int(self.nx) - 1)))

    @traitlets.validate("vmax")
    def _validate_vmax_ge_vmin(self, proposal):
        new_vmax = proposal["value"]
        if new_vmax is not None:
            if not math.isfinite(new_vmax):
                raise traitlets.TraitError(f"vmax must be finite, got {new_vmax}")
            if self.vmin is not None and new_vmax < self.vmin:
                raise traitlets.TraitError(
                    f"vmax ({new_vmax}) must be >= vmin ({self.vmin})"
                )
        return new_vmax

    @traitlets.validate("vmin")
    def _validate_vmin_le_vmax(self, proposal):
        new_vmin = proposal["value"]
        if new_vmin is not None:
            if not math.isfinite(new_vmin):
                raise traitlets.TraitError(f"vmin must be finite, got {new_vmin}")
            if self.vmax is not None and new_vmin > self.vmax:
                raise traitlets.TraitError(
                    f"vmin ({new_vmin}) must be <= vmax ({self.vmax})"
                )
        return new_vmin

    def __init__(
        self,
        data,
        data_b=None,
        *,
        title: str = "",
        title_b: str = "",
        cmap: str = "inferno",
        pixel_size: float = 0.0,
        scale_bar_visible: bool = True,
        z_stretch: float | None = None,
        show_controls: bool = True,
        show_stats: bool = True,
        show_crosshair: bool = False,
        show_fft: bool = False,
        orthographic: bool = False,
        smooth: bool = False,
        flip: bool = False,
        show_diff: bool = False,
        log_scale: bool = False,
        auto_contrast: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        fps: float = 5.0,
        loop: bool = True,
        reverse: bool = False,
        boomerang: bool = False,
        linked_contrast: bool = True,
        play_axis: int = 0,
        dim_labels: list[str] | None = None,
        state=None,
        **kwargs,
    ):
        _reject_unknown_kwargs(type(self), kwargs)
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()
        self.viewer_kind = self._viewer_kind
        # Pre-seed so free() / __repr__ / summary() are safe even if a validator
        # raises before _data is assigned below (e.g. wrong ndim or complex data).
        self._data: np.ndarray | None = None
        self._data_b: np.ndarray | None = None

        # Duck-typed Dataset3d extraction (matches Show2D / Show3D pattern). When the
        # dataset exposes a per-axis sampling tuple [pz, py, px], pass it through as
        # the full anisotropic triple instead of collapsing to a scalar.
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            if not title and data.name:
                title = data.name
            pixel_size_is_default = pixel_size is None or (
                np.isscalar(pixel_size) and float(pixel_size) == 0.0
            )
            if pixel_size_is_default and hasattr(data, "units"):
                try:
                    units = list(data.units)
                    samp = list(data.sampling)
                    # Unit conversion → Å (lateral unit assumed consistent across axes)
                    scale = 10.0 if (units and units[-1] in ("nm",)) else 1.0
                    if units and units[-1] in ("nm", "Å", "angstrom", "A"):
                        if len(samp) >= 3:
                            pixel_size = [float(samp[i]) * scale for i in (-3, -2, -1)]
                        elif len(samp) >= 1:
                            pixel_size = float(samp[-1]) * scale
                except (IndexError, TypeError):
                    pass
            data = data.array

        data = to_numpy(data)
        if data.ndim != 3:
            raise ValueError(f"Show3DVolume requires 3D data, got {data.ndim}D")
        if 0 in data.shape:
            raise ValueError(f"Empty volume: shape {data.shape}. All dims must be >= 1.")
        if not np.isfinite(data).all():
            raise ValueError(
                "Data contains NaN or inf. Clean first: "
                "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
            )
        if np.iscomplexobj(data):
            raise TypeError(
                "Show3DVolume does not accept complex data. Convert first: "
                "np.abs(arr) for magnitude or np.angle(arr) for phase."
            )
        with np.errstate(over="ignore", invalid="ignore"):
            self._data = data.astype(np.float32, copy=False)
        if not np.isfinite(self._data).all():
            raise ValueError(
                "Data exceeds float32 range (|value| > 3.4e38) after cast; "
                "rescale first before passing to Show3DVolume."
            )
        self.nz, self.ny, self.nx = self._data.shape

        # Default to middle slices
        self.slice_z = self.nz // 2
        self.slice_y = self.ny // 2
        self.slice_x = self.nx // 2

        self.title = title
        self.cmap = cmap
        # pixel_size accepts: None → 0 (no scale bar), scalar (isotropic), or
        # 3-tuple/list/ndarray (anisotropic: [pz, py, px] in the same units).
        # For 3-tuple input the scalar trait is set to the lateral mean (py+px)/2
        # so existing scale-bar code keeps working; the full triple is published
        # via pixel_size_axes for per-axis scale bars.
        if pixel_size is None:
            pixel_size = 0.0
        if isinstance(pixel_size, Real) or np.isscalar(pixel_size):
            ps_scalar = float(pixel_size)
            ps_axes = [ps_scalar, ps_scalar, ps_scalar]
        else:
            try:
                ps_axes = [float(v) for v in pixel_size]
            except TypeError:
                raise TypeError(
                    f"pixel_size must be a scalar or 3-element sequence (Å/pixel), "
                    f"got {type(pixel_size).__name__}."
                )
            if len(ps_axes) != 3:
                raise ValueError(
                    f"pixel_size as a sequence must have exactly 3 elements [pz, py, px], "
                    f"got {len(ps_axes)}."
                )
            for v in ps_axes:
                if not math.isfinite(v) or v < 0:
                    raise ValueError(f"pixel_size_axes must be finite and >= 0, got {ps_axes}.")
            ps_scalar = (ps_axes[1] + ps_axes[2]) / 2.0  # lateral mean
        self.pixel_size = ps_scalar
        self.pixel_size_axes = ps_axes
        self.scale_bar_visible = scale_bar_visible
        # Auto-pick z_stretch for thin-Z volumes (e.g. multislice ptycho nz=14, nxy=730).
        thin_z_ratio = min(self.nx, self.ny) / max(self.nz, 1)
        if z_stretch is None:
            # Round to half-step matching slider; clamp to validator range [1, 30].
            z_stretch = max(1.0, min(30.0, round(thin_z_ratio * 2) / 2)) if thin_z_ratio > 4 else 1.0
        self.z_stretch = float(z_stretch)
        self.compact = True
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.show_crosshair = False
        self.show_fft = show_fft
        self.orthographic = orthographic
        self.smooth = smooth
        self.flip = flip
        self.show_diff = show_diff
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.vmin = vmin
        self.vmax = vmax
        self.fps = fps
        self.loop = loop
        self.reverse = reverse
        self.boomerang = boomerang
        self.linked_contrast = linked_contrast
        self.play_axis = play_axis
        if dim_labels is not None:
            self.dim_labels = dim_labels

        # Optional second volume (dual comparison). _data_b pre-seeded above.
        if data_b is not None:
            if hasattr(data_b, "array") and hasattr(data_b, "name") and hasattr(data_b, "sampling"):
                if not title_b and data_b.name:
                    title_b = data_b.name
                data_b = data_b.array
            data_b = to_numpy(data_b)
            if data_b.ndim != 3:
                raise ValueError(f"data_b must be 3D, got {data_b.ndim}D")
            if np.iscomplexobj(data_b):
                raise TypeError(
                    "data_b complex data not accepted. Convert first: "
                    "np.abs(arr) for magnitude or np.angle(arr) for phase."
                )
            if data_b.shape != self._data.shape:
                raise ValueError(
                    f"data_b shape {data_b.shape} must match data shape {self._data.shape}"
                )
            # NaN/inf in data_b silently propagates into stats (UI shows NaN) and breaks
            # the diff panel (|A - B| explodes). Reject up front like primary data.
            if not np.isfinite(data_b).all():
                raise ValueError(
                    "data_b contains NaN or inf. Clean first: "
                    "np.nan_to_num(arr, nan=0, posinf=0, neginf=0)."
                )
            with np.errstate(over="ignore", invalid="ignore"):
                self._data_b = data_b.astype(np.float32, copy=False)
            if not np.isfinite(self._data_b).all():
                raise ValueError(
                    "data_b exceeds float32 range (|value| > 3.4e38) after cast; "
                    "rescale first before passing to Show3DVolume."
                )
            self.dual_mode = True
            self.title_b = title_b
            self.volume_bytes_b = self._data_b.tobytes()

        self._compute_stats()
        self.volume_bytes = self._data.tobytes()
        self.observe(self._on_slice_change, names=["slice_x", "slice_y", "slice_z"])
        self.observe(self._on_playing_change, names=["playing"])
        self.observe(self._on_show_stats_change, names=["show_stats"])
        self.observe(self._on_gif_export, names=["_gif_export_requested"])
        self.observe(self._on_zip_export, names=["_zip_export_requested"])

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                    expected_widget=self._widget_name,
                )
            else:
                state = unwrap_state_payload(state, expected_widget=self._widget_name)
            self.load_state_dict(state)

    def __repr__(self) -> str:
        base = f"{self._widget_name}({self.nz}×{self.ny}×{self.nx}, slices=({self.slice_z},{self.slice_y},{self.slice_x}), cmap={self.cmap}"
        if self.dual_mode:
            base += ", dual=True"
        return base + ")"

    def state_dict(self) -> dict:
        return {
            "title": self.title,
            "viewer_kind": self.viewer_kind,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "show_fft": self.show_fft,
            "orthographic": self.orthographic,
            "smooth": self.smooth,
            "flip": self.flip,
            "pixel_size": self.pixel_size,
            "pixel_size_axes": list(self.pixel_size_axes),
            "scale_bar_visible": self.scale_bar_visible,
            "z_stretch": self.z_stretch,
            "slice_x": self.slice_x,
            "slice_y": self.slice_y,
            "slice_z": self.slice_z,
            "fps": self.fps,
            "loop": self.loop,
            "reverse": self.reverse,
            "boomerang": self.boomerang,
            "play_axis": self.play_axis,
            "dim_labels": list(self.dim_labels),
            "dual_mode": self.dual_mode,
            "title_b": self.title_b,
            "show_diff": self.show_diff,
            "linked_contrast": self.linked_contrast,
        }

    def save(self, path: str) -> None:
        save_state_file(path, self._widget_name, self.state_dict())

    def load_state_dict(self, state: dict) -> None:
        # Surface validator errors. Warn on unknown keys (typo / wrong widget version).
        # Reject dual_mode=True when no data_b — would render an empty B panel.
        if state.get("dual_mode") and self._data_b is None:
            raise ValueError(
                "Saved state has dual_mode=True but this widget was constructed "
                "without data_b. Re-instantiate with data_b before loading."
            )
        allowed = {
            "title", "cmap", "log_scale", "auto_contrast", "vmin", "vmax",
            "viewer_kind",
            "show_stats", "show_controls", "show_crosshair", "show_fft",
            "orthographic", "smooth", "flip", "pixel_size", "pixel_size_axes",
            "scale_bar_visible", "z_stretch", "compact", "slice_x",
            "slice_y", "slice_z", "fps", "loop", "reverse", "boomerang",
            "play_axis", "dim_labels", "title_b", "show_diff",
            "linked_contrast",
        }
        unknown = [k for k in state if k not in allowed and k != "dual_mode"]
        if unknown:
            import warnings
            warnings.warn(
                f"load_state_dict ignored unknown keys: {unknown}. "
                "Likely typo or saved by a different widget version.",
                stacklevel=2,
            )
        state = {k: v for k, v in state.items() if k in allowed}
        state.pop("viewer_kind", None)
        state.pop("show_crosshair", None)
        # Saved states from older versions may include compact=False. The current
        # widget intentionally ignores it and always uses the compact layout.
        state.pop("compact", None)
        vmin_marker = object()
        vmax_marker = object()
        vmin = state.pop("vmin", vmin_marker)
        vmax = state.pop("vmax", vmax_marker)
        if vmin is not vmin_marker or vmax is not vmax_marker:
            new_vmin = self.vmin if vmin is vmin_marker else vmin
            new_vmax = self.vmax if vmax is vmax_marker else vmax
            if new_vmin is not None and new_vmax is not None and float(new_vmin) > float(new_vmax):
                raise traitlets.TraitError(f"vmin ({new_vmin}) must be <= vmax ({new_vmax})")
            # Clear first so either half of a valid saved pair can be loaded
            # regardless of the widget's current contrast limits.
            self.vmin = None
            self.vmax = None
            if new_vmin is not None:
                self.vmin = float(new_vmin)
            if new_vmax is not None:
                self.vmax = float(new_vmax)
        for key, val in state.items():
            if self.has_trait(key):
                setattr(self, key, val)
        self.dual_mode = self._data_b is not None
        # Forward-compat: state saved before pixel_size_axes existed only has the
        # scalar pixel_size. Mirror it across all three axes so depth scale bars
        # don't desync from the lateral one after a load.
        if "pixel_size" in state and "pixel_size_axes" not in state:
            ps = float(state["pixel_size"])
            self.pixel_size_axes = [ps, ps, ps]

    def free(self) -> None:
        """Release RAM held by this widget. `del widget` won't free
        memory because traitlets observers pin the refcount."""
        if self._data is None:
            return
        self._data = None
        self._data_b = None
        for trait in ("volume_bytes", "volume_bytes_b", "_gif_data", "_zip_data"):
            setattr(self, trait, b"")
        import gc
        gc.collect()

    def summary(self) -> None:
        lines = [self.title or self._widget_name, "═" * 32]
        lines.append(f"Volume:   {self.nz}×{self.ny}×{self.nx}")
        if self.pixel_size > 0:
            ps = self.pixel_size
            unit = f"{ps / 10:.2f} nm/px" if ps >= 10 else f"{ps:.2f} Å/px"
            lines[-1] += f" ({unit})"
        labels = list(self.dim_labels)
        lines.append(
            f"Slices:   {labels[0]}={self.slice_z}  {labels[1]}={self.slice_y}  {labels[2]}={self.slice_x}"
        )
        if hasattr(self, "_data") and self._data is not None:
            arr = self._data
            lines.append(
                f"Data:     min={float(arr.min()):.4g}  max={float(arr.max()):.4g}  mean={float(arr.mean()):.4g}"
            )
        if self.dual_mode and self._data_b is not None:
            lines.append(f"Volume B: {self.title_b or 'Volume B'}")
            arr_b = self._data_b
            lines.append(
                f"Data B:   min={float(arr_b.min()):.4g}  max={float(arr_b.max()):.4g}  mean={float(arr_b.mean()):.4g}"
            )
        scale = "log" if self.log_scale else "linear"
        if self.vmin is not None and self.vmax is not None:
            contrast = f"vmin={self.vmin:.4g}, vmax={self.vmax:.4g}"
        elif self.auto_contrast:
            contrast = "auto contrast"
        else:
            contrast = "manual contrast"
        display = f"{self.cmap} | {contrast} | {scale}"
        if self.show_fft:
            display += " | FFT"
        if self.show_diff and self.dual_mode:
            display += " | diff"
        lines.append(f"Display:  {display}")
        print("\n".join(lines))

    def _compute_stats(self) -> None:
        """Compute statistics for the 3 current slices.

        Skipped when show_stats is False to avoid 12 reductions
        per slice movement on multi-MB volumes (JS does not render a stats bar; this
        is for programmatic access only when the caller has opted in).
        """
        if not self.show_stats or self._data is None:
            return
        slices = [
            self._data[self.slice_z, :, :],
            self._data[:, self.slice_y, :],
            self._data[:, :, self.slice_x],
        ]
        with self.hold_sync():
            self.stats_mean = [float(np.mean(s, dtype=np.float64)) for s in slices]
            self.stats_min = [float(np.min(s)) for s in slices]
            self.stats_max = [float(np.max(s)) for s in slices]
            self.stats_std = [float(np.std(s, dtype=np.float64)) for s in slices]
            if self._data_b is not None:
                slices_b = [
                    self._data_b[self.slice_z, :, :],
                    self._data_b[:, self.slice_y, :],
                    self._data_b[:, :, self.slice_x],
                ]
                self.stats_mean_b = [float(np.mean(s, dtype=np.float64)) for s in slices_b]
                self.stats_min_b = [float(np.min(s)) for s in slices_b]
                self.stats_max_b = [float(np.max(s)) for s in slices_b]
                self.stats_std_b = [float(np.std(s, dtype=np.float64)) for s in slices_b]

    def _on_slice_change(self, change) -> None:
        if self.playing:
            return
        self._compute_stats()

    def _on_playing_change(self, change) -> None:
        if not self.playing:
            self._compute_stats()

    def _on_show_stats_change(self, change) -> None:
        if change.get("new"):
            self._compute_stats()

    def play(self) -> Self:
        self.playing = True
        return self

    def pause(self) -> Self:
        self.playing = False
        return self

    def stop(self) -> Self:
        self.playing = False
        self.slice_z = self.nz // 2
        self.slice_y = self.ny // 2
        self.slice_x = self.nx // 2
        return self

    def _on_gif_export(self, change=None) -> None:
        if not self._gif_export_requested:
            return
        self._gif_export_requested = False
        try:
            self._generate_gif()
        except Exception as e:
            # On error: clear _gif_data so JS observer fires and resets
            # exporting=False. Without this the UI shows "..." forever.
            import warnings
            warnings.warn(f"GIF export failed: {type(e).__name__}: {e}")
            self._gif_data = b""

    def _on_zip_export(self, change=None) -> None:
        if not self._zip_export_requested:
            return
        self._zip_export_requested = False
        try:
            self._generate_zip()
        except Exception as e:
            import warnings
            warnings.warn(f"ZIP export failed: {type(e).__name__}: {e}")
            self._zip_data = b""

    def _get_export_slices(self) -> list[np.ndarray]:
        # Pick which volume to export. In dual+show_diff mode, export |A - B|; in dual mode
        # without diff, export Volume A; otherwise Volume A. Single-volume mode always exports A.
        if self.dual_mode and self.show_diff and self._data_b is not None:
            vol = np.abs(self._data.astype(np.float64) - self._data_b.astype(np.float64))
        else:
            vol = self._data
        axis = self._export_axis
        if axis == 0:
            return [vol[z, :, :] for z in range(self.nz)]
        if axis == 1:
            return [vol[:, y, :] for y in range(self.ny)]
        return [vol[:, :, x] for x in range(self.nx)]

    def _normalize_slice(self, slc: np.ndarray) -> np.ndarray:
        if self.log_scale:
            # Signed log so diff frames (in dual mode) don't collapse to zero.
            slc = np.sign(slc) * np.log1p(np.abs(slc))
        # Mirror JS path: when flip=True the on-screen renderer negates the data
        # and flips the contrast range (min<->max with sign). Exports must do the
        # same so saved GIF/ZIP/PNG frames match what the user sees on screen.
        if self.flip:
            slc = -slc
        if self.vmin is not None and self.vmax is not None:
            vmin = float(self.vmin)
            vmax = float(self.vmax)
            if self.log_scale:
                vmin = float(np.sign(vmin) * np.log1p(abs(vmin)))
                vmax = float(np.sign(vmax) * np.log1p(abs(vmax)))
            if self.flip:
                vmin, vmax = -vmax, -vmin
        elif self.auto_contrast:
            vmin = float(np.percentile(slc, 2))
            vmax = float(np.percentile(slc, 98))
        else:
            vmin = float(slc.min())
            vmax = float(slc.max())
        if vmax > vmin:
            return np.clip((slc - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)
        return np.zeros(slc.shape, dtype=np.uint8)

    def _generate_gif(self) -> None:
        import io
        from matplotlib import colormaps
        from PIL import Image

        slices = self._get_export_slices()
        cmap_fn = colormaps.get_cmap(self.cmap)
        pil_frames = []
        for slc in slices:
            normalized = self._normalize_slice(slc)
            rgba = cmap_fn(normalized / 255.0)
            rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
            pil_frames.append(Image.fromarray(rgb))
        if not pil_frames:
            with self.hold_sync():
                self._gif_data = b""
                self._gif_metadata_json = ""
            return
        buf = io.BytesIO()
        duration_ms = int(1000 / max(0.1, self.fps))
        # Shared palette to avoid per-frame quantization flicker.
        pil_p = [f.convert("P", palette=Image.ADAPTIVE, colors=256) for f in pil_frames]
        pil_p[0].save(
            buf, format="GIF", save_all=True, append_images=pil_p[1:],
            duration=duration_ms, loop=0, disposal=2,
        )
        metadata = {
            **build_json_header(self._widget_name),
            "format": "gif",
            "export_kind": "animated_slices",
            "export_axis": int(self._export_axis),
            "n_slices": int(len(pil_frames)),
            "duration_ms": int(duration_ms),
            "display": {
                "cmap": self.cmap,
                "log_scale": bool(self.log_scale),
                "auto_contrast": bool(self.auto_contrast),
            },
        }
        gif_bytes = buf.getvalue()
        # Comm channel chokes on ~50 MB single buffers (VS Code IPC bridge is worse).
        # Warn so the user knows to reduce frame count / resolution if the download stalls.
        if len(gif_bytes) > 50 * 1024 * 1024:
            import warnings
            warnings.warn(
                f"GIF export is {len(gif_bytes) / 1e6:.1f} MB. Jupyter Comm transport "
                f"can stall or fail above ~50 MB. Reduce n_slices or export a smaller "
                f"region if the download does not start.",
                stacklevel=2,
            )
        # Reset to empty first so a re-export with identical bytes still fires the
        # JS trait-change effect (see note in _generate_zip).
        with self.hold_sync():
            self._gif_metadata_json = ""
            self._gif_data = b""
        with self.hold_sync():
            self._gif_metadata_json = json.dumps(metadata, indent=2)
            self._gif_data = gif_bytes

    def _generate_zip(self) -> None:
        import io
        import zipfile
        from matplotlib import colormaps
        from PIL import Image

        slices = self._get_export_slices()
        cmap_fn = colormaps.get_cmap(self.cmap)
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            metadata = {
                **build_json_header(self._widget_name),
                "format": "zip",
                "export_kind": "png_slices",
                "export_axis": int(self._export_axis),
                "n_slices": int(len(slices)),
                "display": {"cmap": self.cmap, "log_scale": bool(self.log_scale)},
            }
            zf.writestr("metadata.json", json.dumps(metadata, indent=2))
            for i, slc in enumerate(slices):
                normalized = self._normalize_slice(slc)
                rgba = cmap_fn(normalized / 255.0)
                rgb = (rgba[:, :, :3] * 255).astype(np.uint8)
                img = Image.fromarray(rgb)
                img_buf = io.BytesIO()
                img.save(img_buf, format="PNG")
                zf.writestr(f"slice_{i:04d}.png", img_buf.getvalue())
        zip_bytes = buf.getvalue()
        if len(zip_bytes) > 50 * 1024 * 1024:
            import warnings
            warnings.warn(
                f"ZIP export is {len(zip_bytes) / 1e6:.1f} MB. Jupyter Comm transport "
                f"can stall or fail above ~50 MB. Consider saving slices individually "
                f"via save_image() if the download does not start.",
                stacklevel=2,
            )
        # Reset to empty first so a re-export with identical bytes still fires the
        # JS trait-change effect (traitlets equality-elides identical Bytes writes).
        # Without this the second click sends the same bytes, JS useEffect on
        # `_zip_data` does not re-run, and the "Exporting..." button stays stuck.
        # Two separate hold_sync blocks force two distinct Comm messages - without
        # this, back-to-back assignments on the same tick coalesce into one.
        with self.hold_sync():
            self._zip_data = b""
        with self.hold_sync():
            self._zip_data = zip_bytes

    def save_image(
        self,
        path: str | pathlib.Path,
        *,
        plane: str | None = None,
        slice_idx: int | None = None,
        format: str | None = None,
        dpi: int = 150,
    ) -> pathlib.Path:
        """Save a volume slice as PNG, PDF, or TIFF.

        Parameters
        ----------
        path : str or pathlib.Path
            Output file path.
        plane : str, optional
            One of 'xy', 'xz', 'yz'. Defaults to 'xy'.
        slice_idx : int, optional
            Slice index along the chosen axis. Defaults to current position.
        format : str, optional
            'png', 'pdf', or 'tiff'. If omitted, inferred from extension.
        dpi : int, default 150
            Output DPI metadata.

        Returns
        -------
        pathlib.Path
            The written file path.
        """
        from matplotlib import colormaps
        from PIL import Image

        path = pathlib.Path(path)
        fmt = (format or path.suffix.lstrip(".").lower() or "png").lower()
        if fmt not in ("png", "pdf", "tiff", "tif"):
            raise ValueError(f"Unsupported format: {fmt!r}. Use 'png', 'pdf', or 'tiff'.")

        plane = (plane or "xy").lower()
        if plane == "xy":
            idx = slice_idx if slice_idx is not None else self.slice_z
            max_idx = self.nz
        elif plane == "xz":
            idx = slice_idx if slice_idx is not None else self.slice_y
            max_idx = self.ny
        elif plane == "yz":
            idx = slice_idx if slice_idx is not None else self.slice_x
            max_idx = self.nx
        else:
            raise ValueError(f"Unknown plane: {plane!r}. Use 'xy', 'xz', or 'yz'.")

        if idx < 0 or idx >= max_idx:
            raise IndexError(f"Slice index {idx} out of range [0, {max_idx}) for plane '{plane}'")

        if plane == "xy":
            slc = self._data[idx]
        elif plane == "xz":
            slc = self._data[:, idx, :]
        else:
            slc = self._data[:, :, idx]

        # Respect dual+show_diff so saved slice matches displayed |A-B|.
        if self.dual_mode and self.show_diff and self._data_b is not None:
            if plane == "xy":
                slc_b = self._data_b[idx]
            elif plane == "xz":
                slc_b = self._data_b[:, idx, :]
            else:
                slc_b = self._data_b[:, :, idx]
            slc = np.abs(slc.astype(np.float64) - slc_b.astype(np.float64))

        normalized = self._normalize_slice(slc)
        cmap_fn = colormaps.get_cmap(self.cmap)
        rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)

        img = Image.fromarray(rgba)
        # PDF requires RGB (no alpha) and the PDF plugin registered.
        if fmt == "pdf":
            Image.init()
            img = img.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        # Pass format explicitly so a mismatched extension (e.g. format="tiff"
        # with path="out.bin") still writes the requested container.
        pil_format = {"png": "PNG", "pdf": "PDF", "tiff": "TIFF", "tif": "TIFF"}[fmt]
        img.save(str(path), format=pil_format, dpi=(dpi, dpi))
        return path
