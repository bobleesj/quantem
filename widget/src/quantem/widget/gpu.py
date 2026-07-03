"""Import-light GPU helpers for notebooks, docs, and live setup checks."""

from __future__ import annotations

import gc
import shutil
import subprocess


def gpu_info() -> list[dict[str, int | str]]:
    """Return a best-effort ``nvidia-smi`` GPU memory snapshot."""
    if shutil.which("nvidia-smi") is None:
        return []
    query = "index,name,memory.used,memory.total,utilization.gpu"
    proc = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        timeout=10,
        check=False,
    )
    if proc.returncode != 0:
        return []

    rows: list[dict[str, int | str]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 5:
            continue
        index, name, used_mib, total_mib, util = parts
        try:
            rows.append(
                {
                    "index": int(index),
                    "name": name,
                    "memory_used_mib": int(used_mib),
                    "memory_total_mib": int(total_mib),
                    "utilization_gpu_percent": int(util),
                }
            )
        except ValueError:
            continue
    return rows


def vram_status() -> str:
    """Return a compact human-readable VRAM summary."""
    rows = gpu_info()
    if not rows:
        return "VRAM: unavailable"
    return "  ".join(
        f"GPU{row['index']} {row['memory_used_mib']}/{row['memory_total_mib']} MiB"
        for row in rows
    )


def free_gpu() -> None:
    """Best-effort release of Python, CuPy, and Torch GPU caches."""
    gc.collect()
    try:
        import cupy as cp  # type: ignore[import-not-found]

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except (ImportError, RuntimeError, AttributeError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except (ImportError, RuntimeError, AttributeError):
        pass


__all__ = ["free_gpu", "gpu_info", "vram_status"]
