"""IO helpers tuned for widget demos.

# Why a separate widget IO module instead of `quantem.core.io.read_2d`

`quantem.core.io.file_readers.read_2d` is the canonical loader and returns
a full `Dataset2d` with calibrated axes, units, and provenance. It routes
through `rsciio.emd.file_reader`, which for Velox EMDs:

1. Opens the HDF5 file
2. Walks the ENTIRE metadata tree (Application, Features, Operations,
   Presentation, Thumbnail, Version, plus every per-Image Metadata JSON
   blob - typically 60 KB of JSON per frame parsed into nested dicts)
3. Builds typed axis objects with calibrated scale/units
4. Decompresses the actual pixel data
5. Returns a dict with `data`, `axes`, `metadata`, ...

For an image-stack widget (Show3D / Show3DSlices) we throw away everything
except `data` + a single pixel size. The metadata expansion + axis
construction in rsciio is ~95% of the wall time:

| Loader (10 x 4096^2 HAADF EMDs) | Wall time | Speedup |
|---------------------------------|-----------|---------|
| rsciio serial (read_2d)         | 10,098 ms | 1.0x    |
| h5py serial (this module)       |    486 ms | 20.8x   |
| h5py thread x8 (this module)    |    308 ms | 32.8x   |

The h5py direct path skips steps 2 + 3 entirely - we open the file, jump
straight to `Data/Image/<uuid>/Data`, pull the pixel array, optionally
read one pixel-size value, close the file. Thread pool wins because HDF5
chunk decompression is CPU-bound and the OS page cache is shared across
threads. Process pool LOSES because pickling 67 MB float32 arrays back
from workers costs more than the parallelism saves (measured: process x8
= 1064 ms, slower than serial threads).

# Scope - narrow on purpose

- EMD metadata extraction is Velox-specific (Thermo Talos/Themis). Other
  EMD producers (Berkeley, NCEM emdfile, py4DSTEM) use different group
  paths + key names and will not surface metadata here. Future versions
  may add dispatch on the EMD subformat.
- Only 2D image planes under `Data/Image/<uuid>/Data` (shape `(H, W, 1)`).
  Spectra, 4D-STEM stacks, and multi-frame EMDs are not handled.
- Returns a `Dataset3d` (from `quantem.core.datastructures`) with extra
  Python attrs (`labels`, `frame_metadata`, `elapsed_ms`) attached.

For full-fidelity loads with axis units, multi-channel detection, and
provenance, use `quantem.core.io.file_readers.read_2d`. For raw stacks
into a viewer, use this module.
"""

from __future__ import annotations

import glob as _glob
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np


_DEFAULT_EXTS = ("emd", "png", "jpg", "tif", "npy")


def _expand_one(p: str) -> list[str]:
    """Handle: explicit file, glob pattern, OR directory (auto-glob for known
    image extensions). Macros macOS resource-fork (`._*`) filter always."""
    path = Path(p)
    if path.is_dir():
        out: list[str] = []
        for ext in _DEFAULT_EXTS:
            out += _glob.glob(str(path / f"*.{ext}"))
        return sorted(q for q in out if "/._" not in q)
    # Either a glob pattern or an explicit file path - glob handles both.
    return sorted(q for q in _glob.glob(p) if "/._" not in q)


def _expand_paths(paths) -> list[str]:
    """Flatten paths input. Accepts a single str/Path (file / glob / dir) OR
    a list of those. Returns deduplicated sorted list of absolute file paths."""
    if isinstance(paths, (str, Path)):
        return _expand_one(str(paths))
    flat: list[str] = []
    seen = set()
    for entry in paths:
        for q in _expand_one(str(entry)):
            if q not in seen:
                seen.add(q)
                flat.append(q)
    return flat


def _ext(path: str) -> str:
    """Lower-case extension without dot. `.tiff` and `.tif` both return `tif`."""
    suffix = Path(path).suffix.lower().lstrip(".")
    return "tif" if suffix == "tiff" else ("jpg" if suffix == "jpeg" else suffix)


def _probe_shape(path: str) -> tuple[int, int] | None:
    """Cheap header-only inspection: first 2D image plane's `(H, W)`.

    Per-format strategy:

    - `.emd` (Velox HDF5): walk `Data/Image/<uuid>/Data` shape.
    - `.png` / `.jpg` / `.tif`: PIL Image opens lazily and reads only the
      header for `.size`.
    - `.npy`: `np.lib.format.read_magic` + header parse without loading data.
    """
    ext = _ext(path)
    try:
        if ext == "emd":
            with h5py.File(path, "r") as fp:
                for uuid in fp["Data/Image"]:
                    dset = fp["Data/Image"][uuid]["Data"]
                    if dset.ndim >= 2:
                        return (int(dset.shape[0]), int(dset.shape[1]))
        elif ext in ("png", "jpg", "tif"):
            from PIL import Image
            with Image.open(path) as img:
                # PIL .size is (W, H) - swap to numpy (H, W) convention.
                return (int(img.size[1]), int(img.size[0]))
        elif ext == "npy":
            with open(path, "rb") as fp:
                version = np.lib.format.read_magic(fp)
                shape, _, _ = np.lib.format._read_array_header(fp, version)
                if len(shape) >= 2:
                    return (int(shape[-2]), int(shape[-1]))
    except (OSError, KeyError, ValueError):
        return None
    return None


def _read_one(path: str, shape: tuple[int, int], dtype: np.dtype) -> tuple[np.ndarray, str, dict] | None:
    """Read one file. Returns `(arr, short_label, metadata)` or `None`.

    `metadata` is a dict per file: for EMD it's the parsed Velox per-image
    JSON (pixel size, voltage, magnification, stage position, detector, ...).
    For PNG/JPG/TIF/NPY there's no embedded metadata so `metadata = {}`.

    EMD: `Data/Image/<uuid>/Data` slice `[..., 0]` (drops singleton frame axis),
    metadata from `Data/Image/<uuid>/Metadata`.
    PNG/JPG/TIF: `np.asarray(PIL.Image.open(path))`, multi-channel → luminance.
    NPY: `np.load(mmap_mode='r')` then materialize the slice.
    """
    ext = _ext(path)
    try:
        arr: np.ndarray | None = None
        meta: dict = {}
        if ext == "emd":
            with h5py.File(path, "r") as fp:
                for uuid in fp["Data/Image"]:
                    grp = fp["Data/Image"][uuid]
                    dset = grp["Data"]
                    if dset.shape[:2] == shape:
                        arr = dset[..., 0]
                        # Parse per-image Velox metadata JSON. Cheap (~60 KB
                        # blob per file) and gives us pixel size, magnification,
                        # voltage, stage tilt, detector type, acquisition date.
                        try:
                            meta_bytes = bytes(grp["Metadata"][:, 0]).rstrip(b"\x00")
                            meta = json.loads(meta_bytes)
                        except (KeyError, ValueError):
                            meta = {}
                        break
        elif ext in ("png", "jpg", "tif"):
            from PIL import Image
            with Image.open(path) as img:
                raw = np.asarray(img)
            if raw.ndim == 3:
                raw = raw[..., :3].mean(axis=-1)
            if raw.shape == shape:
                arr = raw
        elif ext == "npy":
            raw = np.load(path, mmap_mode="r")
            if raw.ndim == 2 and raw.shape == shape:
                arr = np.asarray(raw)
            elif raw.ndim > 2 and raw.shape[-2:] == shape:
                arr = np.asarray(raw[(0,) * (raw.ndim - 2)])
        if arr is None:
            return None
        return arr.astype(dtype, copy=False), Path(path).name[:36], meta
    except (OSError, KeyError, ValueError):
        return None


def _summarize_emd_meta(meta: dict) -> dict:
    """Flatten the parts of a Velox metadata blob a microscopist actually
    cares about into a small dict. Keeps everything else accessible under
    `raw` so power users can dig in."""
    if not meta:
        return {}
    br = meta.get("BinaryResult", {}) or {}
    optics = meta.get("Optics", {}) or {}
    stage = meta.get("Stage", {}) or {}
    acq = meta.get("Acquisition", {}) or {}
    px = br.get("PixelSize", {}) or {}

    def _f(v) -> float | None:
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    px_w = _f(px.get("width"))
    return {
        "pixel_size_A": px_w * 1e10 if px_w is not None else None,
        "pixel_unit": br.get("PixelUnitX", "m"),
        "voltage_kV": (_f(optics.get("AccelerationVoltage")) or 0) / 1000.0 or None,
        "magnification": _f(optics.get("NominalMagnification")),
        "spot_index": optics.get("SpotIndex"),
        "stage_alpha_deg": _f(stage.get("StageTAlpha")),
        "stage_beta_deg": _f(stage.get("StageTBeta")),
        "detector": br.get("DetectorName") or br.get("Detector"),
        "date": acq.get("AcquisitionStartDatetime", {}).get("DateString") if isinstance(acq.get("AcquisitionStartDatetime"), dict) else None,
        "raw": meta,
    }


def _read_pixel_scale_A(path: str) -> float | None:
    """Velox PixelSize is stored in meters in the per-Image JSON metadata
    blob. Convert to Angstroms. Returns `None` if metadata is absent."""
    try:
        with h5py.File(path, "r") as fp:
            for uuid in fp["Data/Image"]:
                meta_bytes = bytes(fp["Data/Image"][uuid]["Metadata"][:, 0]).rstrip(b"\x00")
                meta = json.loads(meta_bytes)
                px = meta.get("BinaryResult", {}).get("PixelSize", {}).get("width")
                if px is None:
                    return None
                return float(px) * 1e10
    except (OSError, KeyError, ValueError):
        return None
    return None


def _load_one_shape(paths: list[str], shape: tuple[int, int], workers: int, dtype: np.dtype, verbose: bool) -> dict:
    """Parallel load + assemble stack for files sharing one `(H, W)`.

    Pre-allocates the output (N, H, W) array and has each thread write its
    frame directly into the right slab. Skips the final `np.stack` copy that
    would otherwise allocate + memcpy the full stack a second time at
    ~80-160 ms per GB of output."""
    t0 = time.perf_counter()
    n = len(paths)
    stack = np.empty((n, shape[0], shape[1]), dtype=dtype)
    labels: list[str | None] = [None] * n
    metas: list[dict] = [{}] * n

    def _read_into(idx_path):
        idx, path = idx_path
        result = _read_one(path, shape, dtype)
        if result is None:
            return idx, None, {}
        arr, label, meta = result
        stack[idx] = arr  # direct write into pre-allocated slab
        return idx, label, meta

    if verbose:
        from tqdm.auto import tqdm
        bar = tqdm(total=n, desc=f"load {shape[0]}x{shape[1]}", unit="file", leave=False)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, label, meta in ex.map(_read_into, enumerate(paths)):
            labels[idx] = label
            metas[idx] = meta
            if verbose:
                bar.update(1)
    if verbose:
        bar.close()

    # Drop any files that didn't match (returned None label). Compact in-place.
    keep = [i for i, lbl in enumerate(labels) if lbl is not None]
    if len(keep) != n:
        stack = stack[keep]
        labels = [labels[i] for i in keep]
        metas = [metas[i] for i in keep]

    # Summarize the per-frame metadata so callers can do e.g.
    # `ds.frame_metadata[5]['voltage_kV']` without parsing raw Velox JSON.
    frame_metadata = [_summarize_emd_meta(m) for m in metas]
    # Use the first frame's pixel size for the Dataset3d sampling. If the
    # stack mixes magnifications, the per-frame value in `frame_metadata` is
    # authoritative; the global sampling is a convenience for scale-bar.
    pixel_size = frame_metadata[0].get("pixel_size_A") if frame_metadata else None
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if verbose:
        nbytes_gb = stack.nbytes / 1e9
        px = f"{pixel_size:.3f} Å/px" if pixel_size else "no pixel-size metadata"
        print(f"loaded {stack.shape[0]} × {shape[0]}×{shape[1]} ({nbytes_gb:.2f} GB, {px}) in {elapsed_ms:.0f} ms")

    # Wrap in quantem.core's Dataset3d so callers can pass it straight to
    # Show3D(ds) and the widget auto-extracts name + sampling + units.
    from quantem.core.datastructures import Dataset3d
    px_lateral = pixel_size if pixel_size is not None else 1.0
    px_units = "Å" if pixel_size is not None else "pixels"
    name = labels[0] if labels else f"image stack {shape[0]}x{shape[1]}"
    ds = Dataset3d.from_array(
        array=stack,
        name=name,
        sampling=[1.0, px_lateral, px_lateral],
        units=["index", px_units, px_units],
    )
    # Attach per-frame labels + metadata + load timing as Python-only attrs.
    # quantem.core's Dataset schema covers axes/sampling/units but doesn't
    # have a per-frame metadata slot; expose ours directly on the instance.
    ds.labels = labels  # type: ignore[attr-defined]
    ds.frame_metadata = frame_metadata  # type: ignore[attr-defined]
    ds.elapsed_ms = elapsed_ms  # type: ignore[attr-defined]
    return ds


def load(paths, *, size: int | tuple[int, int] | None = None, workers: int = 8, dtype: np.dtype = np.float32, verbose: bool = True) -> dict:
    """Load a stack of microscopy images into one numpy array.

    Hand it a folder full of `.emd` / `.png` / `.tif` / `.npy` images
    (or several folders, or a list of files) and it figures out the rest:
    detects the format, reads everything in parallel, groups frames by
    resolution, and returns the data ready for `Show3D`.

    Parameters
    ----------
    paths
        What to load. Accept several flavours:

        - A folder: ``'/data/gold'`` → reads every image inside it
        - A glob pattern: ``'/data/gold/*.emd'`` → matching files
        - A single file: ``'/data/gold/0042.emd'``
        - A list of any of the above:
          ``['/data/run1', '/data/run2', '/data/extra.png']``
        - A numpy array you already have in memory:
          ``np.random.rand(30, 4096, 4096)``

        Supported file types: ``.emd`` (Velox), ``.png``, ``.jpg``,
        ``.tif``, ``.npy``.

    size : int or (height, width), optional
        Only load files at this image resolution. Pass an int for square
        (``size=4096`` ≡ ``size=(4096, 4096)``), or an explicit ``(H, W)``
        tuple for non-square. Useful when a folder contains a mix of
        magnifications and you only want one. Without it, mixed folders
        return a dict-of-dicts keyed by shape.

    workers : int, default 8
        How many files to read at once. 8 works well in most cases.

    dtype : numpy dtype, default ``np.float32``
        Output pixel type. Use ``np.uint16`` to halve memory if your
        detector is 16-bit native (Show3D handles either).

    verbose : bool, default True
        Print a progress bar + one-line summary while loading. Set
        ``False`` for silent scripts. Matches the ``verbose`` flag on
        ``Show4DSTEM`` and other quantem.widget classes.

    Returns
    -------
    dict
        Has keys ``'stack'`` (the (N, H, W) numpy array), ``'labels'``
        (filenames), ``'pixel_size'`` (Å/pixel from EMD metadata,
        ``None`` for other formats), ``'shape'`` (H, W) and
        ``'elapsed_ms'`` (wall-clock load time).

        If the folder contains images at SEVERAL different resolutions,
        the result is instead grouped by shape: ``result[(H, W)]``
        gives the dict above for each resolution.

    Examples
    --------
    Load a folder of gold-nanoparticle EMD images and show them:

    >>> from quantem.widget import load, Show3D
    >>> images = load('/data/gold/v6')                              # doctest: +SKIP
    >>> Show3D(images['stack'],                                     # doctest: +SKIP
    ...        labels=images['labels'],
    ...        pixel_size=images['pixel_size'],
    ...        pixel_unit='A')

    Combine multiple folders into one stack:

    >>> images = load([                                             # doctest: +SKIP
    ...     '/data/run1',
    ...     '/data/run2',
    ...     '/data/extra.png',
    ... ])

    Mixed-resolution folder (different magnifications in one place):

    >>> images = load('/data/mixed_session')                        # doctest: +SKIP
    >>> for shape, group in images.items():                         # doctest: +SKIP
    ...     print(f"{shape}: {len(group['stack'])} frames")
    (4096, 4096): 36 frames
    (1024, 1024): 4 frames
    >>> Show3D(images[(4096, 4096)]['stack'])                       # doctest: +SKIP

    Load a stack of 16-bit PNGs without converting to float32 (saves memory):

    >>> images = load('/data/caitlyn/scan_*.png', dtype='uint16')   # doctest: +SKIP
    >>> Show3D(images['stack'])                                     # doctest: +SKIP

    Show data that's already in memory as a numpy array:

    >>> import numpy as np                                          # doctest: +SKIP
    >>> arr = np.random.rand(30, 4096, 4096).astype('float32')      # doctest: +SKIP
    >>> images = load(arr)                                          # doctest: +SKIP
    >>> Show3D(images['stack'])                                     # doctest: +SKIP

    Silent mode (useful in scripts and CI):

    >>> images = load('/data/*.emd', verbose=False)                 # doctest: +SKIP

    How fast is it
    --------------

    Reads 10 × 4k Velox EMD files in ~300 ms on mjgoat (NVMe SSD,
    8 worker threads). That's ~30× faster than ``quantem.core.io.read_2d``
    because we skip the per-file metadata tree expansion when we just
    need the pixel data.

    Loads 125 × 2048² 16-bit PNGs (the caitlyn drift-corrected stack)
    in ~1.4 s with ``dtype='uint16'``.

    Notes
    -----
    - Pixel scale is read from the first file in each bucket only. Mixed
      magnification within one resolution bucket is not detected here -
      filter by directory or use rsciio for full per-frame metadata.
    - Files whose data group has no 2D plane are silently dropped. Compare
      `sum(len(g["stack"]) for g in result.values())` against the input
      path count to spot any skipped files.
    - For PNG stacks `dtype=np.uint16` saves the float32 cast (~25 % win)
      and halves memory. Show3D handles integer input natively.
    """
    # numpy array passthrough: caller already has a stack in memory, just
    # wrap it in a Dataset3d so downstream code (Show3D, notebooks) treats
    # it identically to a load-from-disk result.
    if isinstance(paths, np.ndarray):
        arr = paths
        if arr.ndim == 2:
            arr = arr[np.newaxis]
        if arr.ndim != 3:
            raise ValueError(f"Array input must be 2D or 3D, got shape {arr.shape}")
        out_arr = arr.astype(dtype, copy=False)
        shape_out = (int(out_arr.shape[1]), int(out_arr.shape[2]))
        from quantem.core.datastructures import Dataset3d
        ds = Dataset3d.from_array(
            array=out_arr,
            name=f"image stack {shape_out[0]}x{shape_out[1]}",
            sampling=[1.0, 1.0, 1.0],
            units=["index", "pixels", "pixels"],
        )
        ds.labels = [str(i) for i in range(out_arr.shape[0])]  # type: ignore[attr-defined]
        ds.frame_metadata = [{} for _ in range(out_arr.shape[0])]  # type: ignore[attr-defined]
        ds.elapsed_ms = 0.0  # type: ignore[attr-defined]
        return ds

    paths = _expand_paths(paths)
    if not paths:
        raise FileNotFoundError(f"No image files matched {paths!r}")

    # Group by resolution via header probe (no pixel data read yet).
    groups: dict[tuple[int, int], list[str]] = {}
    for path in paths:
        shape_probed = _probe_shape(path)
        if shape_probed is None:
            continue
        groups.setdefault(shape_probed, []).append(path)

    if not groups:
        raise FileNotFoundError(f"No 2D image planes found in {len(paths)} files")

    # If caller specified `size`, filter to that resolution and return flat dict.
    # `size=4096` is shorthand for square `(4096, 4096)`.
    if size is not None:
        wanted = (size, size) if isinstance(size, int) else (int(size[0]), int(size[1]))
        if wanted not in groups:
            available = sorted(groups.keys(), key=lambda s: -s[0] * s[1])
            raise ValueError(
                f"No files matched size={wanted}. Available resolutions: {available}"
            )
        return _load_one_shape(groups[wanted], wanted, workers, dtype, verbose)

    # No size filter: single-resolution session returns the inner dict directly
    # so callers write `r['stack']`. Mixed-resolution returns dict-of-dicts.
    buckets = {shape: _load_one_shape(files, shape, workers, dtype, verbose) for shape, files in groups.items()}
    if len(buckets) == 1:
        return next(iter(buckets.values()))
    return buckets
