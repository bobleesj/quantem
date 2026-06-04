"""Shared widget-faithful PNG snapshot helper.

Mirrors the WebGPU pipeline (log_scale -> normalize -> cmap LUT) so the static
PNG looks like the canvas, not a matplotlib-axes copy. Image-only output, no
titles / axes / borders. Used by Show2D / Show3D / Show3DSlices / Show4DSTEM
``_repr_mimebundle_`` static fallback.
"""
from __future__ import annotations

import io
import numpy as np


_VIRIDIS = "viridis"


def _normalize(arr: np.ndarray, vmin: float | None, vmax: float | None,
               log: bool) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if log:
        arr = np.sign(arr) * np.log1p(np.abs(arr))
    if vmin is None or vmax is None:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            if vmin is None:
                vmin = float(np.percentile(finite, 2))
            if vmax is None:
                vmax = float(np.percentile(finite, 98))
        else:
            vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9
    return np.clip((arr - vmin) / (vmax - vmin), 0.0, 1.0)


def _apply_cmap(norm: np.ndarray, cmap: str) -> np.ndarray:
    import matplotlib.cm as cm
    try:
        rgba = cm.get_cmap(cmap)(norm)
    except Exception:
        rgba = cm.get_cmap(_VIRIDIS)(norm)
    return (rgba[..., :3] * 255).astype(np.uint8)


def _downsample_to(arr: np.ndarray, max_px: int) -> np.ndarray:
    h, w = arr.shape[:2]
    if max(h, w) <= max_px:
        return arr
    step = max(h // max_px, w // max_px, 1)
    return arr[::step, ::step]


def render_image_png(img2d: np.ndarray, *,
                     cmap: str = "viridis",
                     vmin: float | None = None,
                     vmax: float | None = None,
                     log: bool = False,
                     max_px: int = 512) -> bytes:
    """Render a single 2D array through normalize -> cmap -> PNG bytes.

    Image-only output (no axes, no title). Mirrors the WebGPU canvas.
    """
    norm = _normalize(img2d, vmin, vmax, log)
    norm = _downsample_to(norm, max_px)
    rgb = _apply_cmap(norm, cmap)
    return _png_bytes(rgb)


def render_panels_png(panels: list[np.ndarray], *,
                      cmaps: list[str] | str = "viridis",
                      ncols: int = 3,
                      max_px_per_panel: int = 256,
                      vmin: float | None = None,
                      vmax: float | None = None,
                      log: bool = False,
                      pad_px: int = 8,
                      pad_value: int = 255,
                      upscale_to_max: bool = True) -> bytes:
    """Tile N panels (possibly different shapes) into a single PNG.

    Each panel normalized + cmapped independently (mirrors widget per-panel
    auto-contrast). Panels rescaled to a common pixel size so small panels
    (e.g. 48x48 CBED next to 256x256 BF) don't swim in whitespace. Image-only.
    """
    if not panels:
        return b""
    if isinstance(cmaps, str):
        cmaps = [cmaps] * len(panels)
    # First pass: downsample large panels, get per-panel RGB
    rgbs: list[np.ndarray] = []
    max_h = max_w = 0
    for img, cm_name in zip(panels, cmaps):
        norm = _normalize(img, vmin, vmax, log)
        norm = _downsample_to(norm, max_px_per_panel)
        rgb = _apply_cmap(norm, cm_name)
        rgbs.append(rgb)
        max_h = max(max_h, rgb.shape[0])
        max_w = max(max_w, rgb.shape[1])
    # Second pass: upscale smaller panels to the largest pair of dims
    # (preserves aspect ratio per panel, no whitespace bloat).
    if upscale_to_max:
        from PIL import Image as _PILImage
        target_max = max(max_h, max_w)
        scaled: list[np.ndarray] = []
        for rgb in rgbs:
            h, w = rgb.shape[:2]
            scale = target_max / max(h, w)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            if new_h == h and new_w == w:
                scaled.append(rgb)
            else:
                im = _PILImage.fromarray(rgb).resize((new_w, new_h),
                                                     _PILImage.NEAREST)
                scaled.append(np.asarray(im))
        rgbs = scaled
        max_h = max(r.shape[0] for r in rgbs)
        max_w = max(r.shape[1] for r in rgbs)
    n = len(rgbs)
    nrows = (n + ncols - 1) // ncols
    canvas = np.full(
        (nrows * max_h + (nrows + 1) * pad_px,
         ncols * max_w + (ncols + 1) * pad_px, 3),
        pad_value, dtype=np.uint8,
    )
    for i, rgb in enumerate(rgbs):
        r, c = divmod(i, ncols)
        y0 = pad_px + r * (max_h + pad_px) + (max_h - rgb.shape[0]) // 2
        x0 = pad_px + c * (max_w + pad_px) + (max_w - rgb.shape[1]) // 2
        h, w = rgb.shape[:2]
        canvas[y0:y0 + h, x0:x0 + w] = rgb
    return _png_bytes(canvas)


def _png_bytes(rgb_u8: np.ndarray) -> bytes:
    from PIL import Image
    img = Image.fromarray(rgb_u8)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()
