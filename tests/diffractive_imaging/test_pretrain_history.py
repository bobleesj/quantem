"""Snapshot capture must not change DIP optimization or retain training graphs."""

import copy

import pytest
import torch

from quantem.core.ml import CNN2d, OptimizerParams
from quantem.diffractive_imaging import ObjectDIP, ProbeDIP


@pytest.mark.parametrize("kind", ["object", "probe"])
def test_capture_preserves_training(kind):
    torch.manual_seed(42)
    dtype = torch.float32 if kind == "object" else torch.complex64
    network = CNN2d(in_channels=2, start_filters=4, num_layers=1, dtype=dtype)
    source = torch.randn(2, 16, 16, dtype=dtype)
    if kind == "object":
        model = ObjectDIP.from_model(
            network, source, num_slices=2, slice_thicknesses=1, obj_type="potential", rng=123
        )
        model.pretrain_target = source.clone()
    else:
        model = ProbeDIP.from_model(network, model_input=source[None], num_probes=2, rng=123)
    captured = copy.deepcopy(model)
    options = dict(num_iters=5, optimizer_params=OptimizerParams.Adam(lr=1e-3), show=False)
    model.pretrain(**options)
    captured.pretrain(**options, snapshot_every=2)
    assert model.pretrain_history is None
    history = captured.pretrain_history
    assert history.iterations == [1, 2, 4, 5]
    assert history.losses == model._pretrain_losses
    assert history.learning_rates == model._pretrain_lrs
    for key, expected in model.model.state_dict().items():
        assert torch.equal(captured.model.state_dict()[key], expected), key
    assert all(value.grad_fn is None for value in history.predictions)
    assert all(value.shape == (16, 16) for value in history.predictions)
    captured.pretrain(**{**options, "num_iters": 1}, snapshot_every=2)
    assert captured.pretrain_history.iterations == [1]
    assert len(captured.pretrain_history.losses) == 1
