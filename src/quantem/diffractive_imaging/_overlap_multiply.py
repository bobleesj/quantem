"""Optional public ptychography tensor operation, with a pure-Torch fallback."""

from collections.abc import Callable
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _overlap_operation() -> Callable:
    """Resolve the optional public API once without hiding import/runtime errors."""
    try:
        from quantem.gpu.ptycho import multiply_overlap
    except ModuleNotFoundError as error:
        if error.name not in {"quantem.gpu", "quantem.gpu.ptycho"}:
            raise
        return torch.mul
    return multiply_overlap


def _multiply_overlap(object_t: torch.Tensor, probe_t: torch.Tensor) -> torch.Tensor:
    """Transmit probe modes through one object slice using the public tensor API."""
    return _overlap_operation()(object_t, probe_t)
