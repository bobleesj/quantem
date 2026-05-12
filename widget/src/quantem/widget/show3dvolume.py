"""
Show3DVolume: orthogonal slice viewer for 3D volumetric data.

Displays XY, XZ, YZ planes with interactive sliders. All slicing happens
in JavaScript for instant response. Useful for ptychography reconstruction
volumes, tomograms, and any voxel data where the user wants three-plane
inspection rather than a frame stack.
"""
import json
import pathlib
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
# js/colormaps.ts COLORMAPS table. Same set as Show3D.
_VALID_CMAPS = frozenset({
    "inferno", "viridis", "plasma", "magma", "hot", "gray",
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
    pixel_size : float, optional
        Pixel size in angstroms for scale bar.
    show_stats : bool, default True
        Show per-slice statistics.
    show_crosshair : bool, default True
        Draw crosshair on each plane showing the other two slices' positions.
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
        Custom labels for the three dimensions. Default ["X", "Y", "Z"].

    Examples
    --------
    >>> import numpy as np
    >>> from quantem.widget import Show3DVolume
    >>> volume = np.random.rand(64, 64, 64).astype(np.float32)
    >>> Show3DVolume(volume, title="My Volume", cmap="viridis")
    """

    _esm = pathlib.Path(__file__).parent / "static" / "show3dvolume.js"
    _css = pathlib.Path(__file__).parent / "static" / "show3dvolume.css"

    widget_version = traitlets.Unicode("unknown").tag(sync=True)

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
    # Stats for volume B (3 values: xy, xz, yz)
    stats_mean_b = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min_b = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max_b = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std_b = traitlets.List(traitlets.Float()).tag(sync=True)
    # Display
    title = traitlets.Unicode("").tag(sync=True)
    cmap = traitlets.Unicode("inferno").tag(sync=True)
    log_scale = traitlets.Bool(False).tag(sync=True)
    auto_contrast = traitlets.Bool(False).tag(sync=True)
    vmin = traitlets.Float(None, allow_none=True).tag(sync=True)
    vmax = traitlets.Float(None, allow_none=True).tag(sync=True)
    # Scale bar
    pixel_size = traitlets.Float(0.0).tag(sync=True)
    scale_bar_visible = traitlets.Bool(True).tag(sync=True)
    # UI
    show_controls = traitlets.Bool(True).tag(sync=True)
    show_stats = traitlets.Bool(True).tag(sync=True)
    show_crosshair = traitlets.Bool(True).tag(sync=True)
    show_fft = traitlets.Bool(False).tag(sync=True)
    # Axis labels (dim 0, 1, 2)
    dim_labels = traitlets.List(traitlets.Unicode(), default_value=["X", "Y", "Z"]).tag(sync=True)
    # Stats (3 values: xy, xz, yz)
    stats_mean = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_min = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_max = traitlets.List(traitlets.Float()).tag(sync=True)
    stats_std = traitlets.List(traitlets.Float()).tag(sync=True)
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
        if val <= 0:
            raise traitlets.TraitError(f"fps must be > 0, got {val}")
        return val

    @traitlets.validate("pixel_size")
    def _validate_pixel_size(self, proposal):
        import math
        val = float(proposal["value"])
        if math.isnan(val) or math.isinf(val):
            raise traitlets.TraitError(f"pixel_size must be finite, got {val}")
        if val < 0:
            raise traitlets.TraitError(f"pixel_size must be >= 0, got {val}")
        return val

    @traitlets.validate("play_axis")
    def _validate_play_axis(self, proposal):
        val = int(proposal["value"])
        if val not in (0, 1, 2, 3):
            raise traitlets.TraitError(f"play_axis must be 0/1/2/3, got {val}")
        return val

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
        if new_vmax is not None and self.vmin is not None and new_vmax < self.vmin:
            raise traitlets.TraitError(
                f"vmax ({new_vmax}) must be >= vmin ({self.vmin})"
            )
        return new_vmax

    @traitlets.validate("vmin")
    def _validate_vmin_le_vmax(self, proposal):
        new_vmin = proposal["value"]
        if new_vmin is not None and self.vmax is not None and new_vmin > self.vmax:
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
        show_controls: bool = True,
        show_stats: bool = True,
        show_crosshair: bool = True,
        show_fft: bool = False,
        show_diff: bool = False,
        log_scale: bool = False,
        auto_contrast: bool = False,
        vmin: float | None = None,
        vmax: float | None = None,
        fps: float = 5.0,
        play_axis: int = 0,
        dim_labels: list[str] | None = None,
        state=None,
        **kwargs,
    ):
        _reject_unknown_kwargs(type(self), kwargs)
        super().__init__(**kwargs)
        self.widget_version = resolve_widget_version()

        # Duck-typed Dataset3d extraction (matches Show2D / Show3D pattern)
        if hasattr(data, "array") and hasattr(data, "name") and hasattr(data, "sampling"):
            if not title and data.name:
                title = data.name
            if pixel_size == 0.0 and hasattr(data, "units"):
                try:
                    units = list(data.units)
                    sampling_val = float(data.sampling[-1])
                    if units[-1] in ("nm",):
                        pixel_size = sampling_val * 10  # nm → Å
                    elif units[-1] in ("Å", "angstrom", "A"):
                        pixel_size = sampling_val
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
        self._data = data.astype(np.float32)
        self.nz, self.ny, self.nx = self._data.shape

        # Default to middle slices
        self.slice_z = self.nz // 2
        self.slice_y = self.ny // 2
        self.slice_x = self.nx // 2

        self.title = title
        self.cmap = cmap
        self.pixel_size = float(pixel_size)
        self.scale_bar_visible = scale_bar_visible
        self.show_controls = show_controls
        self.show_stats = show_stats
        self.show_crosshair = show_crosshair
        self.show_fft = show_fft
        self.show_diff = show_diff
        self.log_scale = log_scale
        self.auto_contrast = auto_contrast
        self.vmin = vmin
        self.vmax = vmax
        self.fps = fps
        self.play_axis = play_axis
        if dim_labels is not None:
            self.dim_labels = dim_labels

        # Optional second volume (dual comparison)
        self._data_b: np.ndarray | None = None
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
            self._data_b = data_b.astype(np.float32)
            self.dual_mode = True
            self.title_b = title_b
            self.volume_bytes_b = self._data_b.tobytes()

        self._compute_stats()
        self.volume_bytes = self._data.tobytes()
        self.observe(self._on_slice_change, names=["slice_x", "slice_y", "slice_z"])
        self.observe(self._on_playing_change, names=["playing"])
        self.observe(self._on_gif_export, names=["_gif_export_requested"])
        self.observe(self._on_zip_export, names=["_zip_export_requested"])

        if state is not None:
            if isinstance(state, (str, pathlib.Path)):
                state = unwrap_state_payload(
                    json.loads(pathlib.Path(state).read_text()),
                    require_envelope=True,
                    expected_widget="Show3DVolume",
                )
            else:
                state = unwrap_state_payload(state, expected_widget="Show3DVolume")
            self.load_state_dict(state)

    def __repr__(self) -> str:
        base = f"Show3DVolume({self.nz}×{self.ny}×{self.nx}, slices=({self.slice_z},{self.slice_y},{self.slice_x}), cmap={self.cmap}"
        if self.dual_mode:
            base += ", dual=True"
        return base + ")"

    def state_dict(self) -> dict:
        return {
            "title": self.title,
            "cmap": self.cmap,
            "log_scale": self.log_scale,
            "auto_contrast": self.auto_contrast,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "show_stats": self.show_stats,
            "show_controls": self.show_controls,
            "show_crosshair": self.show_crosshair,
            "show_fft": self.show_fft,
            "pixel_size": self.pixel_size,
            "scale_bar_visible": self.scale_bar_visible,
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
        }

    def save(self, path: str) -> None:
        save_state_file(path, "Show3DVolume", self.state_dict())

    def load_state_dict(self, state: dict) -> None:
        # Match Show3D: surface validator errors to the user instead of
        # silently dropping fields — easier to debug bad state files.
        for key, val in state.items():
            if hasattr(self, key):
                setattr(self, key, val)

    def summary(self) -> None:
        lines = [self.title or "Show3DVolume", "═" * 32]
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
        """Compute statistics for the 3 current slices."""
        slices = [
            self._data[self.slice_z, :, :],
            self._data[:, self.slice_y, :],
            self._data[:, :, self.slice_x],
        ]
        with self.hold_sync():
            self.stats_mean = [float(np.mean(s)) for s in slices]
            self.stats_min = [float(np.min(s)) for s in slices]
            self.stats_max = [float(np.max(s)) for s in slices]
            self.stats_std = [float(np.std(s)) for s in slices]
            if self._data_b is not None:
                slices_b = [
                    self._data_b[self.slice_z, :, :],
                    self._data_b[:, self.slice_y, :],
                    self._data_b[:, :, self.slice_x],
                ]
                self.stats_mean_b = [float(np.mean(s)) for s in slices_b]
                self.stats_min_b = [float(np.min(s)) for s in slices_b]
                self.stats_max_b = [float(np.max(s)) for s in slices_b]
                self.stats_std_b = [float(np.std(s)) for s in slices_b]

    def _on_slice_change(self, change) -> None:
        if self.playing:
            return
        self._compute_stats()

    def _on_playing_change(self, change) -> None:
        if not self.playing:
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
        self._generate_gif()

    def _on_zip_export(self, change=None) -> None:
        if not self._zip_export_requested:
            return
        self._zip_export_requested = False
        self._generate_zip()

    def _get_export_slices(self) -> list[np.ndarray]:
        axis = self._export_axis
        if axis == 0:
            return [self._data[z, :, :] for z in range(self.nz)]
        if axis == 1:
            return [self._data[:, y, :] for y in range(self.ny)]
        return [self._data[:, :, x] for x in range(self.nx)]

    def _normalize_slice(self, slc: np.ndarray) -> np.ndarray:
        if self.log_scale:
            # Signed log so diff frames (in dual mode) don't collapse to zero.
            slc = np.sign(slc) * np.log1p(np.abs(slc))
        if self.vmin is not None and self.vmax is not None:
            vmin = float(self.vmin)
            vmax = float(self.vmax)
            if self.log_scale:
                vmin = float(np.sign(vmin) * np.log1p(abs(vmin)))
                vmax = float(np.sign(vmax) * np.log1p(abs(vmax)))
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
            **build_json_header("Show3DVolume"),
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
        with self.hold_sync():
            self._gif_metadata_json = json.dumps(metadata, indent=2)
            self._gif_data = buf.getvalue()

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
                **build_json_header("Show3DVolume"),
                "format": "zip",
                "export_kind": "png_slices",
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
        self._zip_data = buf.getvalue()

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
            slc = np.abs(slc - slc_b)

        normalized = self._normalize_slice(slc)
        cmap_fn = colormaps.get_cmap(self.cmap)
        rgba = (cmap_fn(normalized / 255.0) * 255).astype(np.uint8)

        img = Image.fromarray(rgba)
        if fmt == "pdf":
            Image.init()
            img = img.convert("RGB")
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), dpi=(dpi, dpi))
        return path
