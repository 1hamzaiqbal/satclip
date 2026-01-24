"""
RANGE Evaluation Script for Location Encoders

This script evaluates location encoder models on various geospatial tasks including:
- Classification: biome, ecoregion, country, ocean/land
- Regression: temperature, housing prices, elevation, population, ERA5 climate variables

Key parameters are kept consistent with the original RANGE eval for fair comparison:
- Random seed: 42 for train/val splits
- Ridge alpha values: (0.1, 1.0, 10.0)
- Train/val split ratio: 80/20
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import top_k_accuracy_score, make_scorer
from scipy.special import softmax

# ==============================================================================
# Configuration
# ==============================================================================

# Default data directory - update this to your local path
DEFAULT_DATA_DIR = "/projects/bdbk/shared/range_eval_data"

# Random seed for reproducibility (consistent with original RANGE eval)
RANDOM_SEED = 42

# Ridge regression/classification hyperparameters (consistent with original)
RIDGE_ALPHAS = (0.1, 1.0, 10.0)

# Train/val split ratio
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

# ==============================================================================
# Dataset Classes
# ==============================================================================

class BiomeDataset(Dataset):
    """Biome classification dataset from ecoregion data."""

    def __init__(self, data_dir):
        train_path = os.path.join(data_dir, 'ecoregion_train.csv')
        val_path = os.path.join(data_dir, 'ecoregion_val.csv')

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        df = pd.concat([train_df, val_df])
        df = df.dropna(subset=['BIOME_NAME']).reset_index(drop=True)

        self.labels, self.label_map = pd.factorize(df['BIOME_NAME'])
        self.locations = df[['X', 'Y']].values
        self.num_classes = df['BIOME_NAME'].nunique()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = self.labels[idx]
        return loc, label


class EcoregionDataset(Dataset):
    """Ecoregion classification dataset."""

    def __init__(self, data_dir):
        train_path = os.path.join(data_dir, 'ecoregion_train.csv')
        val_path = os.path.join(data_dir, 'ecoregion_val.csv')

        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)
        df = pd.concat([train_df, val_df])
        df = df.dropna(subset=['ECO_NAME']).reset_index(drop=True)

        self.labels, self.label_map = pd.factorize(df['ECO_NAME'])
        self.locations = df[['X', 'Y']].values
        self.num_classes = df['ECO_NAME'].nunique()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = self.labels[idx]
        return loc, label


class CountryDataset(Dataset):
    """Country classification dataset."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['country', 'lat', 'lon']).reset_index(drop=True)

        self.labels, self.label_map = pd.factorize(df['country'])
        self.locations = df[['lon', 'lat']].values
        self.num_classes = df['country'].nunique()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = self.labels[idx]
        return loc, label


class OceanDataset(Dataset):
    """Ocean/land binary classification dataset."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['land', 'lat', 'lon']).reset_index(drop=True)

        self.labels = df['land'].values
        self.locations = df[['lon', 'lat']].values
        self.num_classes = df['land'].nunique()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = self.labels[idx]
        return loc, label


class TemperatureDataset(Dataset):
    """Mean temperature regression dataset."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['meanT']).reset_index(drop=True)

        self.labels = df['meanT'].values
        self.locations = df[['Lon', 'Lat']].values
        self.num_classes = 0  # Regression task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = torch.tensor(self.labels[idx]).double()
        return loc, label


class HousingDataset(Dataset):
    """California housing prices regression dataset."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['median_house_value']).reset_index(drop=True)

        self.labels = df['median_house_value'].values
        self.locations = df[['longitude', 'latitude']].values
        self.num_classes = 0  # Regression task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = torch.tensor(self.labels[idx]).double()
        return loc, label


class ElevationDataset(Dataset):
    """Elevation regression dataset."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['elevation']).reset_index(drop=True)

        self.labels = df['elevation'].values
        self.locations = df[['lon', 'lat']].values
        self.num_classes = 0  # Regression task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = torch.tensor(self.labels[idx]).double()
        return loc, label


class PopulationDataset(Dataset):
    """Population regression dataset (log-transformed)."""

    def __init__(self, data_path):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=['population']).reset_index(drop=True)

        self.labels = df['population'].values
        self.locations = df[['lon', 'lat']].values
        self.num_classes = 0  # Regression task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = torch.tensor(self.labels[idx]).double()
        # Apply log transform: log(1 + population)
        return loc, np.log(1 + label)


class ERA5Dataset(Dataset):
    """ERA5 climate variable regression dataset."""

    def __init__(self, data_path, variable='air_temp_m'):
        df = pd.read_csv(data_path)
        df = df.dropna(subset=[variable]).reset_index(drop=True)

        self.variable = variable
        self.labels = df[variable].values
        self.locations = df[['Longitude', 'Latitude']].values
        self.num_classes = 0  # Regression task

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        loc = torch.from_numpy(self.locations[idx]).double()
        label = torch.tensor(self.labels[idx]).double()
        return loc, label


# ==============================================================================
# Checkerboard Dataset (Synthetic)
# ==============================================================================

def generate_fibonacci_lattice(n_points, n_classes=16):
    """Generate points on a sphere using Fibonacci lattice sampling."""
    import math

    n_points = n_points // 2
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio

    lats, lons, labels = [], [], []

    for i in np.arange(-n_points, n_points):
        lat = np.arcsin((2 * i) / (2 * n_points + 1)) * 180 / np.pi
        lon = (i % phi) * (360 / phi)

        # Wrap longitude to [-180, 180]
        if lon < -180:
            lon += 360
        if lon > 180:
            lon -= 360

        lons.append(lon)
        lats.append(lat)
        labels.append(i % n_classes)

    return np.array(lons), np.array(lats), np.array(labels)


def haversine_distance(lon1, lat1, lon2, lat2, radius=1.0):
    """Calculate pairwise Haversine distances."""
    lon1, lat1 = np.radians(lon1), np.radians(lat1)
    lon2, lat2 = np.radians(lon2), np.radians(lat2)

    dlon = lon2[:, np.newaxis] - lon1
    dlat = lat2[:, np.newaxis] - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2[:, np.newaxis]) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return radius * c


def cart2sph(x, y, z):
    """Convert Cartesian to spherical coordinates."""
    hxy = np.hypot(x, y)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el


def get_checker_data(n_samples, n_support, n_classes, seed=0, grid=False):
    """Generate checkerboard pattern data on a sphere."""
    lons, lats, labels = generate_fibonacci_lattice(n_support, n_classes=n_classes)

    if grid:
        # Use Fibonacci lattice for evaluation grid
        lons_grid, lats_grid, _ = generate_fibonacci_lattice(n_samples)
        distances = haversine_distance(lons_grid, lats_grid, lons, lats)
        labels_grid = labels[distances.argmin(0)]

        lonlats = torch.from_numpy(np.stack([lons_grid, lats_grid])).T
        labels_out = torch.from_numpy(labels_grid)
    else:
        # Random sampling on sphere
        rng = np.random.RandomState(seed)
        x, y, z = rng.normal(size=(3, n_samples))
        az, el = cart2sph(x, y, z)
        lons_seed, lats_seed = np.rad2deg(az), np.rad2deg(el)

        distances = haversine_distance(lons_seed, lats_seed, lons, lats)
        labels_seed = labels[distances.argmin(0)]

        lonlats = torch.from_numpy(np.stack([lons_seed, lats_seed])).T
        labels_out = torch.from_numpy(labels_seed)

    return lonlats, torch.zeros_like(labels_out), labels_out


class CheckerboardDataset:
    """Synthetic checkerboard classification dataset on a sphere."""

    def __init__(self, n_samples=10000, n_classes=16, n_support=200):
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.n_support = n_support

        # Training set (seed=0 for consistency)
        self.train_ds = TensorDataset(*get_checker_data(
            n_samples=n_samples, n_support=n_support, n_classes=n_classes, seed=0
        ))

        # Validation set (seed=1 for consistency)
        self.valid_ds = TensorDataset(*get_checker_data(
            n_samples=n_samples, n_support=n_support, n_classes=n_classes, seed=1
        ))

        # Evaluation set (grid-based)
        self.eval_ds = TensorDataset(*get_checker_data(
            n_samples=n_samples, n_support=n_support, n_classes=n_classes, grid=True
        ))


# ==============================================================================
# Dataset Loading
# ==============================================================================

# Task type mapping
CLASSIFICATION_TASKS = {'biome', 'ecoregion', 'country', 'ocean'}
REGRESSION_TASKS = {'temperature', 'housing', 'elevation', 'population'}


def get_dataset(task_name, data_dir=DEFAULT_DATA_DIR, batch_size=256, num_workers=4):
    """
    Load dataset for a given task.

    Args:
        task_name: Name of the evaluation task
        data_dir: Directory containing evaluation data
        batch_size: Batch size for data loaders
        num_workers: Number of workers for data loading

    Returns:
        train_loader, val_loader, num_classes, task_type
    """
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    if task_name == 'biome':
        dataset = BiomeDataset(data_dir)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = dataset.num_classes
        task_type = 'classification'

    elif task_name == 'ecoregion':
        dataset = EcoregionDataset(data_dir)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = dataset.num_classes
        task_type = 'classification'

    elif task_name == 'country':
        data_path = os.path.join(data_dir, 'country.csv')
        dataset = CountryDataset(data_path)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = dataset.num_classes
        task_type = 'classification'

    elif task_name == 'ocean':
        train_path = os.path.join(data_dir, 'land_ocean_train.csv')
        val_path = os.path.join(data_dir, 'land_ocean_test.csv')
        train_ds = OceanDataset(train_path)
        val_ds = OceanDataset(val_path)
        num_classes = train_ds.num_classes
        task_type = 'classification'

    elif task_name == 'temperature':
        data_path = os.path.join(data_dir, 'temp.csv')
        dataset = TemperatureDataset(data_path)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = 0
        task_type = 'regression'

    elif task_name == 'housing':
        data_path = os.path.join(data_dir, 'housing.csv')
        dataset = HousingDataset(data_path)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = 0
        task_type = 'regression'

    elif task_name == 'elevation':
        data_path = os.path.join(data_dir, 'elevation.csv')
        dataset = ElevationDataset(data_path)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = 0
        task_type = 'regression'

    elif task_name == 'population':
        data_path = os.path.join(data_dir, 'population.csv')
        dataset = PopulationDataset(data_path)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = 0
        task_type = 'regression'

    elif task_name.startswith('era5-'):
        # ERA5 tasks: era5-air_temp_m, era5-precip_m, etc.
        variable = task_name.split('-', 1)[1]
        data_path = os.path.join(data_dir, 'ERA5_Land_Clipped_2020.csv')
        dataset = ERA5Dataset(data_path, variable=variable)
        train_ds, val_ds = random_split(dataset, [TRAIN_RATIO, VAL_RATIO], generator=generator)
        num_classes = 0
        task_type = 'regression'

    elif task_name.startswith('checker_'):
        # Checkerboard tasks: checker_100, checker_200, etc.
        n_support = int(task_name.split('_')[1])
        n_classes = 16
        checker = CheckerboardDataset(n_samples=10000, n_classes=n_classes, n_support=n_support)
        train_ds = checker.train_ds
        val_ds = checker.eval_ds
        num_classes = n_classes
        task_type = 'classification'

    else:
        raise ValueError(f"Unknown task: {task_name}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, num_workers=num_workers,
        shuffle=False, drop_last=False
    )

    return train_loader, val_loader, num_classes, task_type


# ==============================================================================
# Embedding Extraction
# ==============================================================================

def extract_embeddings(data_loader, model, device='cuda'):
    """
    Extract embeddings from a location encoder model.

    Args:
        data_loader: DataLoader providing (coords, labels) pairs
        model: Location encoder model
        device: Device to run inference on

    Returns:
        embeddings: numpy array of shape (N, embedding_dim)
        labels: numpy array of shape (N,)
    """
    model.eval()
    model = model.to(device)

    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings"):
            coords, labels = batch
            coords = coords.to(device)

            # Handle models that may return numpy directly
            try:
                emb = model(coords).cpu().numpy()
            except AttributeError:
                emb = model(coords)

            embeddings_list.append(emb)
            labels_list.append(labels.numpy())

    embeddings = np.concatenate(embeddings_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    return embeddings, labels


def save_embeddings(embeddings, labels, save_path):
    """Save embeddings and labels to npz file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, embeddings=embeddings, labels=labels)
    print(f"Saved embeddings to {save_path}")


def load_embeddings(load_path):
    """Load embeddings and labels from npz file."""
    data = np.load(load_path, allow_pickle=True)
    return data['embeddings'], data['labels']


# ==============================================================================
# Evaluation
# ==============================================================================

def evaluate_embeddings(train_embeddings, train_labels, val_embeddings, val_labels, task_type):
    """
    Evaluate embeddings using Ridge regression/classification.

    Args:
        train_embeddings: Training set embeddings
        train_labels: Training set labels
        val_embeddings: Validation set embeddings
        val_labels: Validation set labels
        task_type: 'classification' or 'regression'

    Returns:
        score: Accuracy (classification) or R² (regression)
        model: Fitted Ridge model
    """
    # Normalize embeddings
    scaler = MinMaxScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    val_embeddings = scaler.transform(val_embeddings)

    if task_type == 'classification':
        model = RidgeClassifierCV(alphas=RIDGE_ALPHAS, cv=10)
    else:
        model = RidgeCV(alphas=RIDGE_ALPHAS, cv=3)

    model.fit(train_embeddings, train_labels)
    score = model.score(val_embeddings, val_labels)

    return score, model


def evaluate_task(task_name, model, data_dir=DEFAULT_DATA_DIR, device='cuda',
                  batch_size=256, num_workers=4):
    """
    Full evaluation pipeline for a single task.

    Args:
        task_name: Name of the evaluation task
        model: Location encoder model
        data_dir: Directory containing evaluation data
        device: Device to run inference on
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading

    Returns:
        score: Evaluation metric (accuracy or R²)
    """
    # Load data
    train_loader, val_loader, num_classes, task_type = get_dataset(
        task_name, data_dir, batch_size, num_workers
    )

    print(f"\nEvaluating task: {task_name}")
    print(f"  Task type: {task_type}")
    if task_type == 'classification':
        print(f"  Num classes: {num_classes}")

    # Extract embeddings
    train_emb, train_labels = extract_embeddings(train_loader, model, device)
    val_emb, val_labels = extract_embeddings(val_loader, model, device)

    print(f"  Train samples: {len(train_labels)}")
    print(f"  Val samples: {len(val_labels)}")

    # Evaluate
    score, _ = evaluate_embeddings(train_emb, train_labels, val_emb, val_labels, task_type)

    metric_name = "Accuracy" if task_type == 'classification' else "R²"
    print(f"  {metric_name}: {score:.4f}")

    return score


def evaluate_all_tasks(model, data_dir=DEFAULT_DATA_DIR, device='cuda',
                       batch_size=256, num_workers=4, tasks=None):
    """
    Evaluate model on all available tasks.

    Args:
        model: Location encoder model
        data_dir: Directory containing evaluation data
        device: Device to run inference on
        batch_size: Batch size for data loading
        num_workers: Number of workers for data loading
        tasks: List of tasks to evaluate (None = all available)

    Returns:
        results: Dictionary mapping task names to scores
    """
    if tasks is None:
        tasks = [
            # Classification tasks
            'biome', 'ecoregion', 'country', 'ocean',
            # Regression tasks
            'temperature', 'housing', 'elevation', 'population',
            # Checkerboard tasks
            'checker_100', 'checker_200',
        ]

    results = {}
    for task in tasks:
        try:
            score = evaluate_task(
                task, model, data_dir, device, batch_size, num_workers
            )
            results[task] = score
        except Exception as e:
            print(f"Error evaluating {task}: {e}")
            results[task] = None

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)

    for task, score in results.items():
        if score is not None:
            print(f"  {task:20s}: {score:.4f}")
        else:
            print(f"  {task:20s}: FAILED")

    return results


# ==============================================================================
# Model Loading
# ==============================================================================

def load_location_encoder(checkpoint_path: str, device: str = 'cuda'):
    """Load a location encoder from a checkpoint.

    Supports:
    - Lightning checkpoints (.ckpt) from LearnedActivationsModule or ContrastiveLearningModule
    - Raw state dict checkpoints (.pt, .pth)

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model on

    Returns:
        model: Location encoder model ready for inference
    """
    from experiments.models.location_encoder import LocationEncoder

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if this is a Lightning checkpoint (has 'hyper_parameters' or 'state_dict')
    if isinstance(checkpoint, dict) and 'hyper_parameters' in checkpoint:
        hparams = checkpoint['hyper_parameters']
        state_dict = checkpoint['state_dict']

        # Determine which module type based on hparams
        if 'vision_encoder' in hparams:
            # ContrastiveLearningModule - extract location_encoder
            print("Detected ContrastiveLearningModule checkpoint")

            # Build location encoder from hparams
            model = LocationEncoder(
                encoding_type=hparams.get('encoding_type', 'spherical_harmonics'),
                encoding_config=hparams.get('encoding_config', {}),
                hidden_dim=hparams.get('hidden_dim', 256),
                output_dim=hparams.get('embed_dim', 256),
                num_layers=hparams.get('num_layers', 2),
                activation_type=hparams.get('activation_type', 'siren'),
                activation_config=hparams.get('activation_config', {}),
            )

            # Extract location_encoder weights from state_dict
            loc_state_dict = {}
            prefix = 'location_encoder.'
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    loc_state_dict[k[len(prefix):]] = v

            model.load_state_dict(loc_state_dict)
            print(f"Loaded location encoder with {sum(p.numel() for p in model.parameters())} parameters")

        else:
            # LearnedActivationsModule - extract the inner model
            print("Detected LearnedActivationsModule checkpoint")

            model = LocationEncoder(
                encoding_type=hparams.get('encoding_type', 'spherical_harmonics'),
                encoding_config=hparams.get('encoding_config', {}),
                hidden_dim=hparams.get('hidden_dim', 256),
                output_dim=hparams.get('num_classes', 256),
                num_layers=hparams.get('num_layers', 3),
                activation_type=hparams.get('activation_type', 'relu'),
                activation_config=hparams.get('activation_config', {}),
            )

            # Extract model weights (LocationRegressor contains LocationEncoder)
            loc_state_dict = {}
            prefix = 'model.encoder.'
            for k, v in state_dict.items():
                if k.startswith(prefix):
                    loc_state_dict[k[len(prefix):]] = v

            if loc_state_dict:
                model.load_state_dict(loc_state_dict)
            else:
                # Try direct model loading
                prefix = 'model.'
                for k, v in state_dict.items():
                    if k.startswith(prefix):
                        loc_state_dict[k[len(prefix):]] = v
                model.load_state_dict(loc_state_dict, strict=False)

            print(f"Loaded model with {sum(p.numel() for p in model.parameters())} parameters")

    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Simple state dict checkpoint with config
        print("Detected state_dict checkpoint")

        config = checkpoint.get('config', {})
        model = LocationEncoder(**config)
        model.load_state_dict(checkpoint['state_dict'])

    elif isinstance(checkpoint, dict):
        # Raw state dict - try to infer config from keys
        print("Detected raw state dict checkpoint")
        print("Warning: Attempting to infer model config from state dict keys")

        # Create default model and try to load
        model = LocationEncoder()
        model.load_state_dict(checkpoint, strict=False)

    else:
        raise ValueError(f"Unknown checkpoint format: {type(checkpoint)}")

    model = model.to(device)
    model.eval()

    return model


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="RANGE Evaluation for Location Encoders")

    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR,
                        help='Directory containing evaluation data')
    parser.add_argument('--tasks', type=str, nargs='+', default=None,
                        help='Tasks to evaluate (default: all)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save results')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_location_encoder(args.model_path, args.device)

    # Run evaluation
    results = evaluate_all_tasks(
        model, args.data_dir, args.device, args.batch_size, args.num_workers, args.tasks
    )

    # Save results
    if args.output_dir:
        import json
        os.makedirs(args.output_dir, exist_ok=True)
        results_path = os.path.join(args.output_dir, 'eval_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
