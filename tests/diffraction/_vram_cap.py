"""Simulate a smaller GPU by capping free VRAM, to prove a real workload fits.

The only honest way to know a full-size workload fits a target card is to RUN it
with only that much VRAM free and confirm it does not OOM. A memory ESTIMATE is a
guess - it misses transients (cast slabs, FFT pairs, per-batch buffers), and an
auto-sizing workload (batch size, chunking) picks a DIFFERENT, larger config when
it sees a big card, so a 96 GB run never exercises the 24/48 GB path. Cap the VRAM,
run the real thing, look at whether it survives.
"""

import contextlib
import warnings

import torch


@contextlib.contextmanager
def vram_capped(target_gb: float, device: str = "cuda:0"):
    """Occupy the GPU so only ``target_gb`` of VRAM stays free; yield the cap bytes.

    The workload inside the ``with`` sees ``target_gb`` free, so its auto-sizing
    picks the small-card configuration - the only way to surface an out-of-memory
    that a large card would hide. The yielded value is the number of bytes occupied
    by the cap block; subtract it from ``torch.cuda.max_memory_allocated`` to get
    the workload's own peak (the cap is allocated before the peak stats are reset,
    so it otherwise counts as baseline).

    Parameters
    ----------
    target_gb : float
        Free VRAM to leave, in GB. This is the simulated card size MINUS whatever
        is already resident on entry, so enter this AFTER loading the data the
        workload keeps live (e.g. after alignment, before the merge). If you cap
        before loading, the load itself eats into ``target_gb``.
    device : str
        CUDA device to cap.

    Examples
    --------
    >>> maped.real_space_align(...)                 # resident data loaded first
    >>> with vram_capped(48) as cap_bytes:          # only 48 GB free now
    ...     merged = maped.merge_datasets()         # must not raise OutOfMemoryError
    >>> peak_gb = (torch.cuda.max_memory_allocated(0) - cap_bytes) / 1e9
    """
    dev = torch.device(device)
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info(dev)
    occupy = int(free - target_gb * 1e9)
    if occupy <= 0:
        # Already at or below target free: the cap cannot simulate a card SMALLER
        # than reality, so the workload runs against the real free VRAM. Warn loudly
        # so a passing run is not mistaken for a verified fit at target_gb.
        warnings.warn(
            f"vram_capped: only {free / 1e9:.0f} GB free, cannot cap to "
            f"{target_gb:.0f} GB; workload runs against real free VRAM, not the cap.",
            stacklevel=2,
        )
        occupy = 0
    block = torch.empty(occupy, dtype=torch.uint8, device=dev) if occupy else None
    torch.cuda.reset_peak_memory_stats(dev)
    try:
        yield occupy
    finally:
        del block
        torch.cuda.empty_cache()


def fits_in_vram(target_gb: float, run, device: str = "cuda:0") -> tuple[bool, float]:
    """Run ``run()`` with only ``target_gb`` free; return ``(fits, peak_gb)``.

    ``run`` is a zero-argument callable doing the REAL, full-size workload (full
    data, no binning). Returns ``(True, peak_gb)`` if it completed - ``peak_gb`` is
    the workload's own measured peak (cap subtracted), the definitive "this card
    needs N GB" number - or ``(False, target_gb)`` if it raised
    ``torch.cuda.OutOfMemoryError`` (does not fit the target).

    Examples
    --------
    >>> fits, peak = fits_in_vram(48, lambda: maped.merge_datasets())
    >>> assert fits, "merge OOM'd at 48 GB"
    >>> assert peak < 48, f"peak {peak:.0f} GB over target"
    """
    dev = torch.device(device)
    try:
        with vram_capped(target_gb, device) as cap_bytes:
            run()
        peak_gb = (torch.cuda.max_memory_allocated(dev) - cap_bytes) / 1e9
        return True, peak_gb
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return False, float(target_gb)
