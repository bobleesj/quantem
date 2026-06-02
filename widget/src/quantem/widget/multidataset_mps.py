"""Lazy multi-dataset MacBook viewer: see dataset 0 in ~2s, browse the rest as
they decode behind a slider.

``load_4dstem_macbook(masters)`` decodes dataset 0 synchronously (~1.7s at bin4),
builds ONE Show4DSTEM_MACBOOK whose 5D frame slider spans ALL N datasets, then a
single background GPU-worker thread decodes datasets 1..N-1 into the live
container. The user browses immediately; sliding to a not-yet-decoded dataset
shows the last ready one until its slot fills (auto-updates). A progress line
prints ``[k/N loaded]`` as each finishes.

One dedicated worker owns every Metal decode (the command queue is serial — one
owner is the safe + correct model; verified MPS decode runs off the main thread).
Memory is the same as loading all upfront (~1.2 GB each at bin4); lazy hides the
TIME, not the footprint. Run ``io.survey(folder)`` first to confirm it all fits.

Usage::

    from quantem.widget.multidataset_mps import load_4dstem_macbook
    viewer = load_4dstem_macbook(master_paths, det_bin=4)
    viewer    # dataset 0 shows now; slide the frame axis across datasets
"""
from __future__ import annotations

import os
import threading
import time


def load_4dstem_macbook(masters, *, det_bin: int = 4, scan_size: int | None = None,
                        verbose: bool = True, **viewer_kwargs):
    """Decode dataset 0, show a 5D viewer over all N, fill 1..N-1 in background.

    ``masters`` is either a folder (every ``*_master.h5`` in it is discovered +
    sorted, no hardcoding) or an explicit list of master paths. ``scan_size``
    (e.g. 512 or 256) keeps only masters whose scan is that NxN size - a mixed
    folder holding both 512 and 256 acquisitions is filtered to one, so the 5D
    stack is uniform. Reads HDF5 headers only, no decode.
    """
    from quantem.widget.io import discover_masters, load
    from quantem.widget.kernels.compute.mps import ChunkedFrames, MultiChunkedFrames
    from quantem.widget.show4dstem_mps import Show4DSTEM_MACBOOK

    # folder -> auto-discover (optionally filtered to one scan size); list -> as given
    if isinstance(masters, (str, os.PathLike)) and os.path.isdir(os.path.expanduser(str(masters))):
        scan_shape = (int(scan_size), int(scan_size)) if scan_size else None
        masters = discover_masters(os.path.expanduser(str(masters)),
                                   scan_shape=scan_shape, verbose=False)
    masters = [str(m) for m in masters]
    n = len(masters)
    if n == 0:
        raise ValueError("no master files found")
    names = [os.path.basename(m)[:-len("_master.h5")]
             if m.endswith("_master.h5") else os.path.basename(m) for m in masters]

    def _decode(path):
        # load() returns a LoadResult(data, meta); data is the MPSChunked4DSTEM
        # (chunks + metadata). Wrap in the compute container so MultiChunkedFrames
        # sees a uniform ChunkedFrames.
        data, _meta = load(path, backend="mps", det_bin=det_bin, verbose=False)
        row_prefix = bool(getattr(data, "row_prefix", False)
                          or getattr(data, "metadata", {}).get("row_prefix", False))
        return ChunkedFrames(data, row_prefix=row_prefix)

    def _say(msg):
        if verbose:
            print(msg, flush=True)

    _say(f"[1/{n}] loading {names[0]} ...")
    _t0 = time.perf_counter()
    ds0 = _decode(masters[0])
    _say(f"[1/{n}] {names[0]} ready in {time.perf_counter() - _t0:.1f}s")
    multi = MultiChunkedFrames([ds0], n_total=n, names=names)

    # Build the viewer FIRST so its on_ready hook is wired before any background
    # decode lands. The widget TITLE also carries progress (current file name +
    # "loading k/N" tail that clears when done); the stdout lines below mirror it
    # in the cell output so the operator can watch either.
    viewer = Show4DSTEM_MACBOOK(multi, verbose=viewer_kwargs.pop("verbose", False),
                               **viewer_kwargs)

    if n > 1:
        def _worker():
            for i in range(1, n):
                try:
                    _say(f"[{i + 1}/{n}] loading {names[i]} ...")
                    _t = time.perf_counter()
                    multi.set_dataset(i, _decode(masters[i]))
                    _say(f"[{i + 1}/{n}] {names[i]} ready in {time.perf_counter() - _t:.1f}s")
                except Exception as exc:
                    _say(f"[{i + 1}/{n}] {names[i]} FAILED: {str(exc)[:80]}")
        threading.Thread(target=_worker, daemon=True).start()

    return viewer
