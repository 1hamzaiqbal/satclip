"""Coordinate utilities for converting between lat/lon and unit sphere vectors."""
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import math


def latlon_to_sphere(coords: torch.Tensor) -> torch.Tensor:
    """Convert lat/lon coordinates to 3D unit sphere vectors.

    Args:
        coords: Tensor of shape (N, 2) with (lat, lon) in degrees.
                lat in [-90, 90], lon in [-180, 180]

    Returns:
        Tensor of shape (N, 3) with unit vectors (x, y, z) on S².
    """
    lat = torch.deg2rad(coords[:, 0])
    lon = torch.deg2rad(coords[:, 1])

    x = torch.cos(lat) * torch.cos(lon)
    y = torch.cos(lat) * torch.sin(lon)
    z = torch.sin(lat)

    return torch.stack([x, y, z], dim=-1)


def sphere_to_latlon(xyz: torch.Tensor) -> torch.Tensor:
    """Convert 3D unit sphere vectors to lat/lon coordinates.

    Args:
        xyz: Tensor of shape (N, 3) with unit vectors on S².

    Returns:
        Tensor of shape (N, 2) with (lat, lon) in degrees.
    """
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]

    lat = torch.asin(z.clamp(-1, 1))
    lon = torch.atan2(y, x)

    return torch.rad2deg(torch.stack([lat, lon], dim=-1))


def coordinate_jitter(coords: torch.Tensor, radius: float = 0.01) -> torch.Tensor:
    """Add small random jitter to coordinates for data augmentation.

    Args:
        coords: Tensor of shape (N, 2) with (lat, lon) in degrees.
        radius: Maximum jitter in degrees (~1km at equator for 0.01).

    Returns:
        Jittered coordinates.
    """
    jitter = (torch.rand_like(coords) * 2 - 1) * radius
    return coords + jitter


def load_sites(path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Load Voronoi sites from .npz file.

    Args:
        path: Path to .npz file containing:
            - sites: (n_sites, 3) normalized unit vectors
            - embeddings: (optional) (n_sites, embed_dim) initial embeddings

    Returns:
        Tuple of (sites, embeddings) as torch tensors.
        embeddings is None if not present in file.
    """
    data = np.load(path)

    sites = torch.tensor(data["sites"], dtype=torch.float32)

    embeddings = None
    if "embeddings" in data:
        embeddings = torch.tensor(data["embeddings"], dtype=torch.float32)

    return sites, embeddings


def load_sites_from_coords(path: str, coord_format: str = "latlon") -> torch.Tensor:
    """Load sites from a file containing coordinates.

    Converts coordinates to normalized unit sphere vectors.

    Args:
        path: Path to .npz or .npy file with coordinates.
        coord_format: "latlon" for (lat, lon) degrees, "xyz" for 3D vectors.

    Returns:
        (n_sites, 3) tensor of normalized unit vectors.
    """
    data = np.load(path)

    # Handle .npz vs .npy
    if isinstance(data, np.lib.npyio.NpzFile):
        key = list(data.keys())[0]
        coords = data[key]
    else:
        coords = data

    coords = torch.tensor(coords, dtype=torch.float32)

    if coord_format == "latlon":
        sites = latlon_to_sphere(coords)
    else:
        sites = F.normalize(coords, dim=-1)

    return sites
