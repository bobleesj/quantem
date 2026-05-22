"""No-op profile stub.

Why: source profile.py was deleted but notebooks still call `profile()`.
Stub keeps notebooks runnable. Replace with real profiler when needed.
"""
from __future__ import annotations

import warnings


class _ProfileResult:
    def __init__(self):
        self.records: list = []

    def __repr__(self) -> str:
        return ""


_warned = False


def profile(*args, **kwargs) -> _ProfileResult:
    global _warned
    if not _warned:
        warnings.warn(
            "quantem.widget.profile is a no-op stub. Calls do nothing.",
            stacklevel=2,
        )
        _warned = True
    return _ProfileResult()
