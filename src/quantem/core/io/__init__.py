from quantem.core.io.serialize import AutoSerialize as AutoSerialize
from quantem.core.io.serialize import load as load
from quantem.core.io.serialize import print_file as print_file


# file_readers is imported lazily because its module-level Dataset imports
# would create a cycle with quantem.core.datastructures (datastructures →
# dataset.py → io.serialize → io → file_readers → datastructures).
def __getattr__(name):
    if name in (
        "read_2d",
        "read_4dstem",
        "read_emdfile_to_4dstem",
    ):
        from quantem.core.io import file_readers
        return getattr(file_readers, name)
    raise AttributeError(f"module 'quantem.core.io' has no attribute {name!r}")
