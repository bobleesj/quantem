"""Ptychography keeps working with or without the optional GPU public API."""

import builtins
from types import SimpleNamespace

import pytest
import torch

from quantem.diffractive_imaging import _overlap_multiply as overlap


@pytest.fixture(autouse=True)
def clear_optional_operation_cache():
    overlap._overlap_operation.cache_clear()
    yield
    overlap._overlap_operation.cache_clear()


def _inputs():
    generator = torch.Generator().manual_seed(75)
    object_t = torch.randn(3, 12, 10, dtype=torch.complex128, generator=generator)
    probe_t = torch.randn(2, 3, 12, 10, dtype=torch.complex128, generator=generator)
    return object_t, probe_t


def _assert_transmission_and_gradients():
    object_t, probe_t = _inputs()
    outputs = []
    for multiply in (torch.mul, overlap._multiply_overlap):
        current_object_t = object_t.transpose(-1, -2).detach().requires_grad_()
        current_probe_t = probe_t.transpose(-1, -2).conj().detach().requires_grad_()
        waves_t = multiply(current_object_t, current_probe_t)
        gradients = torch.autograd.grad(
            waves_t.abs().square().sum(), (current_object_t, current_probe_t)
        )
        outputs.append((waves_t, *gradients))
    for expected_t, actual_t in zip(*outputs):
        expected_t = expected_t.detach().resolve_conj().resolve_neg().contiguous()
        actual_t = actual_t.detach().resolve_conj().resolve_neg().contiguous()
        assert torch.equal(
            expected_t.reshape(-1).view(torch.uint8), actual_t.reshape(-1).view(torch.uint8)
        )


def test_public_backend_preserves_transmission_and_gradients(monkeypatch):
    """The reconstruction delegates resident tensors to the public GPU operation."""
    real_import = builtins.__import__
    calls = []

    def multiply(object_t, probe_t):
        calls.append((object_t.device, probe_t.device))
        return torch.mul(object_t, probe_t)

    def import_public(name, *args, **kwargs):
        if name == "quantem.gpu.ptycho":
            return SimpleNamespace(multiply_overlap=multiply)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_public)
    _assert_transmission_and_gradients()
    assert calls == [(torch.device("cpu"), torch.device("cpu"))]


@pytest.mark.parametrize("missing", ["quantem.gpu", "quantem.gpu.ptycho"])
def test_optional_install_preserves_torch_workflow(monkeypatch, missing):
    """An absent GPU package or new API retains native waves and gradients."""
    real_import = builtins.__import__

    def without_public(name, *args, **kwargs):
        if name == "quantem.gpu.ptycho":
            raise ModuleNotFoundError(name=missing)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_public)
    _assert_transmission_and_gradients()


def test_dependency_import_failure_is_not_hidden(monkeypatch):
    """A broken installed dependency is reported instead of silently changing execution."""
    real_import = builtins.__import__

    def broken_public(name, *args, **kwargs):
        if name == "quantem.gpu.ptycho":
            raise ModuleNotFoundError(name="broken_dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_public)
    with pytest.raises(ModuleNotFoundError) as failure:
        overlap._multiply_overlap(*_inputs())
    assert failure.value.name == "broken_dependency"


def test_public_operation_failure_is_not_hidden(monkeypatch):
    """An installed backend's runtime error remains visible to the scientist."""
    real_import = builtins.__import__

    def broken_operation(object_t, probe_t):
        raise RuntimeError("backend execution failed")

    def import_public(name, *args, **kwargs):
        if name == "quantem.gpu.ptycho":
            return SimpleNamespace(multiply_overlap=broken_operation)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_public)
    with pytest.raises(RuntimeError, match="backend execution failed"):
        overlap._multiply_overlap(*_inputs())
