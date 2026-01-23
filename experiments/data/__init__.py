# Data package for experiments
"""
Data loading utilities for learned activations experiments:
- HuggingFace SatCLIP dataset (RGB from davanstrien/satclip)
- HuggingFace Multispectral dataset (local Arrow format, 13-band Sentinel-2)
- Geographic data (elevation, population)
- Synthetic data (checkerboard)
- Coordinate utilities
"""

from .huggingface import SatCLIPHuggingFaceDataModule
from .hf_datamodule import HFMultispectralDataModule, HFSatelliteDataset
from .geographic import ElevationDataset, PopulationDataset, GeographicDataModule
from .synthetic import CheckerboardDataset, SyntheticDataModule
from .coord_utils import latlon_to_sphere, sphere_to_latlon, coordinate_jitter, load_sites

__all__ = [
    # HuggingFace datasets
    "SatCLIPHuggingFaceDataModule",
    "HFMultispectralDataModule",
    "HFSatelliteDataset",
    # Geographic
    "ElevationDataset",
    "PopulationDataset",
    "GeographicDataModule",
    # Synthetic
    "CheckerboardDataset",
    "SyntheticDataModule",
    # Coordinate utilities
    "latlon_to_sphere",
    "sphere_to_latlon",
    "coordinate_jitter",
    "load_sites",
]
