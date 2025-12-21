# SatCLIP Effective Resolution Investigation

## Overview

This document organizes the research plan for investigating the **effective resolution** of SatCLIP's location encoder, specifically comparing models with different Legendre polynomial degrees (L=10 vs L=40).

---

## Background: How SatCLIP Location Encoding Works

### Spherical Harmonics & The L Parameter

SatCLIP uses **spherical harmonics** to encode geographic coordinates (lon/lat) into high-dimensional embeddings:

```
Location (lon, lat) → Spherical Harmonics → Neural Network → 256-dim embedding
```

The **L parameter** (legendre_polys) controls the maximum degree of spherical harmonics:
- **L=10**: Embedding dimension = 10² = 100 features → Lower spatial resolution
- **L=40**: Embedding dimension = 40² = 1600 features → Higher spatial resolution

### Intuition
- Higher L = more Legendre polynomials = finer frequency components = ability to represent smaller geographic features
- Lower L = smoother, coarser representations = may blur fine-grained spatial patterns

### Available Pretrained Models

| Model | Vision Encoder | L | HuggingFace Path |
|-------|---------------|---|------------------|
| satclip-resnet18-l10 | ResNet18 | 10 | microsoft/SatCLIP-ResNet18-L10 |
| satclip-resnet18-l40 | ResNet18 | 40 | microsoft/SatCLIP-ResNet18-L40 |
| satclip-resnet50-l10 | ResNet50 | 10 | microsoft/SatCLIP-ResNet50-L10 |
| satclip-resnet50-l40 | ResNet50 | 40 | microsoft/SatCLIP-ResNet50-L40 |
| satclip-vit16-l10 | ViT-S/16 | 10 | microsoft/SatCLIP-ViT16-L10 |
| satclip-vit16-l40 | ViT-S/16 | 40 | microsoft/SatCLIP-ViT16-L40 |

---

## Research Questions (from meeting notes)

1. **What is the actual effective resolution of SatCLIP?**
   - How well can L=10 vs L=40 distinguish nearby locations?
   - At what spatial scale does performance degrade?

2. **Does higher L improve performance on fine-grained tasks?**
   - Ecoregions Level 1-3 (progressively finer classification)
   - Census block group population density (high-resolution US data)

3. **Checkerboard test idea:**
   - Create synthetic classification task with checkerboard pattern
   - Vary checkerboard cell size (e.g., 100km, 10km, 1km squares)
   - See at what resolution each model fails to distinguish cells

---

## Experiment Plan

### Phase 1: Verify Setup Works (Colab Test)

**Goal**: Confirm we can load both L=10 and L=40 models and get embeddings

See: `00_satclip_test.ipynb` (created below)

### Phase 2: Embedding Similarity Analysis

**Hypothesis**: L=40 should produce more different embeddings for nearby locations than L=10

**Method**:
1. Generate grid of locations at varying distances (1km, 10km, 100km, 1000km apart)
2. Compute embeddings with both L=10 and L=40
3. Calculate cosine similarity between neighboring points
4. Plot similarity vs distance curves for both models

**Expected result**: L=40 should show lower similarity (more discrimination) at fine scales

### Phase 3: Ecoregion Classification

**Task**: Predict ecoregion level from location embeddings

**Levels** (from coarse to fine):
- Level 1: ~14 biomes globally (e.g., Tropical Forest, Desert)
- Level 2: ~50 regions
- Level 3: ~100+ sub-regions

**Method**:
1. Get ecoregion dataset with labels at all 3 levels
2. Train simple classifier (Linear or MLP) on SatCLIP embeddings
3. Compare L=10 vs L=40 accuracy at each level

**Expected result**: L=40 should outperform L=10 especially at finer levels

### Phase 4: Checkerboard Resolution Test

**Synthetic task to directly measure effective resolution**

**Method**:
1. Create synthetic binary classification: assign each location to class 0 or 1 based on checkerboard pattern
2. Vary checkerboard cell size: 500km, 100km, 50km, 10km, 5km, 1km
3. Train classifier on SatCLIP embeddings
4. Find the cell size where accuracy drops to ~50% (random) = effective resolution limit

**This directly answers**: "What is the smallest feature size SatCLIP can reliably distinguish?"

### Phase 5: Census Block Population Density (Stretch Goal)

**High-resolution real-world task**

**Method**:
1. Get US Census block group data with population density
2. Sample locations within block groups
3. Predict population density from SatCLIP embeddings
4. Compare L=10 vs L=40 R² scores

---

## Key Code Patterns

### Loading Models

```python
from huggingface_hub import hf_hub_download
import sys
sys.path.append('./satclip')
from load import get_satclip
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load L=10 model
model_l10 = get_satclip(
    hf_hub_download("microsoft/SatCLIP-ViT16-L10", "satclip-vit16-l10.ckpt"),
    device=device,
)
model_l10.eval()

# Load L=40 model
model_l40 = get_satclip(
    hf_hub_download("microsoft/SatCLIP-ViT16-L40", "satclip-vit16-l40.ckpt"),
    device=device,
)
model_l40.eval()
```

### Getting Embeddings

```python
# coords shape: (N, 2) with [longitude, latitude] in degrees
coords = torch.tensor([
    [-122.4, 37.8],   # San Francisco
    [-74.0, 40.7],    # New York
    [0.1, 51.5],      # London
]).double()

with torch.no_grad():
    emb_l10 = model_l10(coords.to(device)).cpu()  # Shape: (N, 256)
    emb_l40 = model_l40(coords.to(device)).cpu()  # Shape: (N, 256)
```

### Computing Similarity

```python
import torch.nn.functional as F

def cosine_similarity(emb1, emb2):
    return F.cosine_similarity(emb1, emb2, dim=-1)

# Compare embeddings of two nearby points
sim = cosine_similarity(emb_l10[0:1], emb_l10[1:2])
```

---

## Relevant Paper Insights (from SatCLIP paper)

From the benchmarking table in meeting notes:
- SatCLIP outperforms baselines on most tasks (Air Temp, Elevation, Pop Density, Countries, Ecoregions)
- Performance varies by continent (Asia vs Africa)
- Ecoregion prediction is harder than Biome prediction (finer-grained)

Key theoretical point: **Spherical harmonics provide a natural basis for representing functions on the sphere**, with higher L capturing higher-frequency variations.

---

## Data Sources

1. **Ecoregions**: WWF Ecoregions dataset - https://www.worldwildlife.org/publications/terrestrial-ecoregions-of-the-world
2. **Census Blocks**: US Census Bureau - https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html
3. **WorldClim** (for temperature validation): https://www.worldclim.org/

---

## Next Steps

1. [ ] Run `00_satclip_test.ipynb` in Colab to verify setup
2. [ ] Implement embedding similarity analysis
3. [ ] Download ecoregion shapefiles
4. [ ] Implement checkerboard test
5. [ ] Run full comparison experiments

---

## Notes & Ideas

- Consider also testing intermediate L values (L=20, L=30) if time permits
- The "multi-scale RFF" mentioned in meeting notes refers to Random Fourier Features - an alternative to spherical harmonics
- Could visualize embeddings with t-SNE/UMAP colored by geographic region
