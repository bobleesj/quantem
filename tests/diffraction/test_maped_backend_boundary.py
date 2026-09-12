"""MAPED keeps accelerator-native implementation details in quantem.gpu."""

import ast
from pathlib import Path

import quantem.diffraction.maped as maped_module


def test_maped_has_no_native_cuda_dependencies() -> None:
    """MAPED may use Torch CUDA, but not CuPy, NVML, or raw CUDA kernels."""
    source = Path(maped_module.__file__).read_text()
    tree = ast.parse(source)
    imported_roots = set()
    native_kernel_attributes = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
        elif isinstance(node, ast.Attribute) and node.attr in {"RawKernel", "RawModule"}:
            native_kernel_attributes.add(node.attr)

    assert imported_roots.isdisjoint({"cupy", "pynvml"})
    assert not native_kernel_attributes
