"""MAPED owns Torch science while quantem.gpu supplies generic infrastructure."""

import ast
from pathlib import Path

import quantem.diffraction.maped as maped_module


def test_maped_has_no_native_accelerator_dependencies() -> None:
    """MAPED may use Torch accelerators, but not native backend libraries."""
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

    assert imported_roots.isdisjoint({"cupy", "mlx", "pynvml"})
    assert not native_kernel_attributes


def test_maped_uses_only_public_quantem_gpu_modules() -> None:
    """Cross-package calls use stable QuantEM.GPU contracts."""
    source = Path(maped_module.__file__).read_text()
    tree = ast.parse(source)
    private_imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("quantem.gpu._")
    }
    assert not private_imports


def test_resident_maped_implementation_is_torch_only() -> None:
    """The bounded MAPED algorithm stays in QuantEM and uses no native backend."""
    source = Path(maped_module.__file__).with_name("_maped_resident.py").read_text()
    tree = ast.parse(source)
    imported_roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots.isdisjoint({"cupy", "mlx", "pynvml"})
