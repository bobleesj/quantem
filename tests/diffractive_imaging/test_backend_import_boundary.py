"""Keep native execution owned by quantem.gpu, not the ptychography algorithm."""

import ast
from pathlib import Path


def test_diffractive_imaging_has_no_native_runtime_imports():
    """Scientific algorithms use Torch or public GPU APIs across the backend boundary."""
    root = Path(__file__).resolve().parents[2] / "src/quantem/diffractive_imaging"
    native = {"triton", "cupy", "pycuda", "pynvml", "mlx", "Metal", "cuda"}
    violations = []
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            elif (
                isinstance(node, ast.Call)
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and ast.unparse(node.func).split(".")[-1] in {"import_module", "__import__"}
            ):
                modules = [node.args[0].value]
            for module in modules:
                if module.split(".")[0] in native:
                    violations.append(f"{path.name}:{node.lineno}: {module}")
    assert not violations, "\n".join(violations)
