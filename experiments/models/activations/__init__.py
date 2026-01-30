# Activation functions package
"""
Modular activation functions for location encoders:
- ReLU (baseline)
- SIREN (sinusoidal)
- Spline (learned)
- RFF (Random Fourier Features)
- MoE Spline (mixture-of-experts spline)
- Gated Spline (GLU-style gated spline)
"""

from .relu import ReLUActivation
from .siren import SIRENActivation, Sine
from .spline import SplineActivation
from .rff import RFFActivation
from .moe_spline import MoESplineActivation
from .gated_spline import GatedSplineActivation

__all__ = [
    "ReLUActivation",
    "SIRENActivation",
    "Sine",
    "SplineActivation",
    "RFFActivation",
    "MoESplineActivation",
    "GatedSplineActivation",
]

# Registry for easy config-based instantiation
ACTIVATION_REGISTRY = {
    "relu": ReLUActivation,
    "siren": SIRENActivation,
    "spline": SplineActivation,
    "rff": RFFActivation,
    "moe_spline": MoESplineActivation,
    "gated_spline": GatedSplineActivation,
}


def get_activation(name: str, **kwargs):
    """Get activation class by name.

    Args:
        name: Activation name (relu, siren, spline, rff)
        **kwargs: Activation-specific parameters

    Returns:
        Instantiated activation module
    """
    if name not in ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation: {name}. Available: {list(ACTIVATION_REGISTRY.keys())}")
    return ACTIVATION_REGISTRY[name](**kwargs)
