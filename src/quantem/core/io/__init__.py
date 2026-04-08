"""quantem.core.io package.

``file_readers`` imports ``Dataset`` from ``quantem.core.datastructures``,
and ``datastructures.dataset`` imports ``AutoSerialize`` from this module.
If this ``__init__`` eagerly imported ``file_readers``, that chain becomes
circular: ``datastructures/__init__`` → ``dataset`` → ``io/__init__`` →
``file_readers`` → ``datastructures/Dataset`` (still being initialized).

To break the cycle we only eagerly expose the serialize symbols that
``datastructures.dataset`` actually needs at import time. The file-reader
symbols are exposed lazily via ``__getattr__`` so that importing them
first fully initializes ``datastructures`` before ``file_readers`` asks
for ``Dataset``.
"""

from quantem.core.io.serialize import AutoSerialize as AutoSerialize
from quantem.core.io.serialize import load as load
from quantem.core.io.serialize import print_file as print_file

_LAZY_NAMES = {
    "read_2d",
    "read_4dstem",
    "read_emdfile_to_4dstem",
}


def __getattr__(name):
    if name in _LAZY_NAMES:
        from quantem.core.io import file_readers

        value = getattr(file_readers, name)
        globals()[name] = value  # cache for subsequent access
        return value
    raise AttributeError(f"module 'quantem.core.io' has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_LAZY_NAMES))
