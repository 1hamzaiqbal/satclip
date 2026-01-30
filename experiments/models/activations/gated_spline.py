"""Gated Spline Activation Function.

GLU-style gating applied to a learnable spline activation:
    output = spline(x) * sigmoid(gate_spline(x))

The gate spline learns an input-dependent mask that selectively
suppresses or amplifies the primary spline's output per-element.
This is simpler than MoE (no routing network) but still allows
input-dependent activation behavior.
"""

import torch
import torch.nn as nn
from typing import Literal, Tuple

from .spline import SplineActivation


class GatedSplineActivation(nn.Module):
    """Gated spline activation: spline(x) * sigmoid(gate_spline(x)).

    Two independent splines:
    - Primary spline: learns the activation function
    - Gate spline: learns a soft mask (passed through sigmoid)

    The gate enables input-dependent activation strength, allowing
    the network to selectively apply the spline transformation.
    """

    def __init__(
        self,
        n_knots: int = 15,
        input_range: Tuple[float, float] = (-3.0, 3.0),
        init: Literal["relu", "linear", "zero", "tanh", "gelu"] = "relu",
        gate_init: Literal["relu", "linear", "zero", "tanh", "gelu"] = "linear",
        learnable_positions: bool = False,
        **kwargs,
    ):
        """Initialize gated spline activation.

        Args:
            n_knots: Number of knot points per spline
            input_range: (min, max) range for knot positions
            init: Initialization for primary spline (recommended: "relu")
            gate_init: Initialization for gate spline (recommended: "linear")
            learnable_positions: Whether knot x-positions are learnable
            **kwargs: Ignored (for interface consistency)
        """
        super().__init__()
        self.n_knots = n_knots
        self.input_range = input_range
        self.init_type = init
        self.gate_init_type = gate_init

        # Primary spline: learns the activation shape
        self.spline = SplineActivation(
            n_knots=n_knots,
            input_range=input_range,
            init=init,
            learnable_positions=learnable_positions,
        )

        # Gate spline: learns the gating function
        self.gate = SplineActivation(
            n_knots=n_knots,
            input_range=input_range,
            init=gate_init,
            learnable_positions=learnable_positions,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gated spline activation.

        Args:
            x: Input tensor of any shape

        Returns:
            Activated tensor of same shape
        """
        return self.spline(x) * torch.sigmoid(self.gate(x))

    def get_knot_data(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """Get knot data for both splines.

        Returns:
            Tuple of (primary_knots, gate_knots), each being (knot_x, knot_y)
        """
        return self.spline.get_knot_data(), self.gate.get_knot_data()

    def extra_repr(self) -> str:
        return (
            f"n_knots={self.n_knots}, "
            f"input_range={self.input_range}, "
            f"init={self.init_type}, "
            f"gate_init={self.gate_init_type}"
        )
