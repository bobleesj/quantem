"""MAPED owns Torch science while quantem.gpu supplies generic infrastructure."""

import ast
from pathlib import Path

import pytest

import quantem.diffraction.maped as maped


@pytest.mark.parametrize("filename", ["maped.py", "_maped_resident.py"])
def test_maped_has_no_native_accelerator_dependencies(filename: str) -> None:
    """Both public orchestration and bounded merging retain Torch ownership."""
    tree = ast.parse(Path(maped.__file__).with_name(filename).read_text())
    imports = set()
    native_kernels = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Attribute) and node.attr in {"RawKernel", "RawModule"}:
            native_kernels.add(node.attr)

    native_roots = {"cupy", "pycuda", "mlx", "pynvml", "Metal", "objc", "ctypes"}
    assert not {name for name in imports if name.split(".")[0] in native_roots}
    assert not native_kernels
    private_gpu_imports = {
        name for name in imports
        if name.startswith("quantem.gpu.")
        and any(part.startswith("_") or part == "backends"
                for part in name.split(".")[2:])
    }
    assert not private_gpu_imports
