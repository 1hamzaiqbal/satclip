"""Mixture-of-Experts Spline Activation Function.

Multiple expert splines with input-dependent routing. Each expert is
initialized with a different activation shape (relu, tanh, gelu, linear),
allowing the network to select different activation patterns per input.

This addresses the limitation of a single spline activation: smooth regions
and sharp geographic boundaries can use different expert activations.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

from .spline import SplineActivation


class MoESplineActivation(nn.Module):
    """Mixture-of-Experts spline activation with input-dependent routing.

    Architecture:
        Input x → Router(x) → gate weights (n_experts,)
                → Expert_0(x), Expert_1(x), ..., Expert_K(x)
                → Σ gate[i] * Expert_i(x)

    Each expert is a SplineActivation with a different initialization,
    encouraging specialization into different activation patterns.
    """

    def __init__(
        self,
        n_experts: int = 4,
        n_knots: int = 15,
        input_range: Tuple[float, float] = (-3.0, 3.0),
        expert_inits: Optional[List[str]] = None,
        load_balance_weight: float = 0.0,
        learnable_positions: bool = False,
        **kwargs,
    ):
        """Initialize MoE spline activation.

        Args:
            n_experts: Number of expert splines
            n_knots: Number of knot points per expert
            input_range: (min, max) range for knot positions
            expert_inits: List of initialization strategies per expert.
                Defaults to ["relu", "tanh", "gelu", "linear"] (truncated/cycled to n_experts).
            load_balance_weight: Weight for load balancing auxiliary loss (0 = disabled)
            learnable_positions: Whether knot x-positions are learnable
            **kwargs: Ignored (for interface consistency)
        """
        super().__init__()
        self.n_experts = n_experts
        self.n_knots = n_knots
        self.input_range = input_range
        self.load_balance_weight = load_balance_weight

        # Determine expert initializations
        default_inits = ["relu", "tanh", "gelu", "linear"]
        if expert_inits is None:
            expert_inits = default_inits
        # Cycle if fewer inits than experts
        while len(expert_inits) < n_experts:
            expert_inits.append(default_inits[len(expert_inits) % len(default_inits)])
        expert_inits = expert_inits[:n_experts]
        self.expert_inits = expert_inits

        # Create expert splines
        self.experts = nn.ModuleList([
            SplineActivation(
                n_knots=n_knots,
                input_range=input_range,
                init=init,
                learnable_positions=learnable_positions,
            )
            for init in expert_inits
        ])

        # Router: lazy initialization (hidden_dim unknown at construction)
        self.router = nn.LazyLinear(n_experts, bias=False)

        # Track expert usage for logging
        self.register_buffer("_expert_counts", torch.zeros(n_experts))
        self.register_buffer("_total_calls", torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MoE spline activation.

        Args:
            x: Input tensor of shape (batch, hidden_dim)

        Returns:
            Activated tensor of same shape
        """
        # Compute routing weights
        gate_logits = self.router(x)  # (batch, n_experts)
        gates = torch.softmax(gate_logits, dim=-1)  # (batch, n_experts)

        # Apply all experts: each returns (batch, hidden_dim)
        expert_outputs = torch.stack(
            [expert(x) for expert in self.experts], dim=-1
        )  # (batch, hidden_dim, n_experts)

        # Weighted combination
        # gates: (batch, n_experts) -> (batch, 1, n_experts)
        output = (gates.unsqueeze(1) * expert_outputs).sum(dim=-1)
        # (batch, hidden_dim)

        # Track expert usage (no grad)
        if self.training:
            with torch.no_grad():
                self._expert_counts += gates.sum(dim=0).detach()
                self._total_calls += x.shape[0]

        return output

    def get_load_balance_loss(self) -> torch.Tensor:
        """Compute load balancing auxiliary loss.

        Encourages all experts to be used equally. Returns 0 if
        load_balance_weight is 0.

        Returns:
            Scalar loss tensor
        """
        if self.load_balance_weight == 0.0:
            return torch.tensor(0.0, device=self._expert_counts.device)

        if self._total_calls == 0:
            return torch.tensor(0.0, device=self._expert_counts.device)

        # Fraction of tokens routed to each expert
        fractions = self._expert_counts / self._total_calls
        # Ideal uniform distribution
        target = 1.0 / self.n_experts
        # Squared deviation from uniform
        loss = self.load_balance_weight * self.n_experts * (fractions * fractions).sum()

        return loss

    def reset_usage_stats(self):
        """Reset expert usage tracking (call at epoch boundaries)."""
        self._expert_counts.zero_()
        self._total_calls.zero_()

    def get_expert_usage(self) -> torch.Tensor:
        """Get expert usage fractions for logging.

        Returns:
            Tensor of shape (n_experts,) with usage fractions
        """
        if self._total_calls == 0:
            return torch.ones(self.n_experts) / self.n_experts
        return (self._expert_counts / self._total_calls).detach().cpu()

    def get_knot_data(self) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Get knot data from all experts for visualization.

        Returns:
            List of (knot_x, knot_y) tuples, one per expert
        """
        return [expert.get_knot_data() for expert in self.experts]

    def extra_repr(self) -> str:
        return (
            f"n_experts={self.n_experts}, "
            f"n_knots={self.n_knots}, "
            f"input_range={self.input_range}, "
            f"expert_inits={self.expert_inits}, "
            f"load_balance_weight={self.load_balance_weight}"
        )
