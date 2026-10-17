"""Optional, device-resident display snapshots of DIP pretraining."""

from dataclasses import dataclass, field

import torch


@dataclass
class PretrainHistory:
    """Predictions evaluated during one pretraining call, averaged over axis 0.

    Images remain on the training device until a viewer exports them. Iterations
    are one-based evaluations before their optimizer update. Learning rates are
    recorded after the scheduler step, matching the existing training plots.
    """

    target: torch.Tensor
    kind: str
    obj_type: str = "potential"
    iterations: list[int] = field(default_factory=list)
    predictions: list[torch.Tensor] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)

    def record(
        self,
        output: torch.Tensor,
        loss: float,
        learning_rate: float,
        iteration: int,
        total: int,
        every: int,
    ) -> None:
        """Record metrics and selected means without retaining autograd graphs."""
        self.losses.append(loss)
        self.learning_rates.append(learning_rate)
        if iteration == 1 or iteration % every == 0 or iteration == total:
            self.iterations.append(iteration)
            self.predictions.append(output.detach().mean(0).clone())
