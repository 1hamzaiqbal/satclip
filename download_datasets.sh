#!/bin/bash
# =============================================================================
# SatCLIP Dataset Download Script
# =============================================================================
# This script downloads all datasets used in the SatCLIP research notebooks.
# Run with: chmod +x download_datasets.sh && ./download_datasets.sh
# =============================================================================

set -e  # Exit on any error

echo "============================================================================="
echo "SatCLIP Dataset Download Script"
echo "============================================================================="
echo ""

# Create directories
mkdir -p datasets/models
mkdir -p datasets/elevation
mkdir -p datasets/population
mkdir -p datasets/ecoregions
mkdir -p datasets/natural_earth
mkdir -p datasets/temperature

# =============================================================================
# 1. GLOBAL AIR TEMPERATURE DATASET (~375 KB)
# =============================================================================
# Source: Springer Nature Figshare
# Paper: "Global Air Temperature" (Scientific Data, Nature)
# Format: CSV with columns [longitude, latitude, elevation, ..., temperature, ...]
# Resolution: Point observations at weather stations worldwide
# Use case: Benchmark regression task for location encoders
# Citation: Used in SatCLIP paper (Section 4.1) for temperature prediction
# =============================================================================
echo ""
echo "[1/6] Downloading Global Air Temperature Dataset..."
echo "      Source: Springer Nature Figshare"
echo "      Size: ~375 KB"
echo "      Format: CSV (lon, lat, elevation, temperature readings)"
curl -L -o datasets/temperature/temperature.csv \
    "https://springernature.figshare.com/ndownloader/files/12609182"
echo "      Done: datasets/temperature/temperature.csv"

# =============================================================================
# 2. ETOPO 2022 GLOBAL ELEVATION DATA (~478 MB)
# =============================================================================
# Source: NOAA National Centers for Environmental Information (NCEI)
# Dataset: ETOPO 2022 - Earth TOPOgraphy 2022
# Format: NetCDF (.nc) - self-describing scientific data format
# Resolution: 60 arc-seconds (~1.85 km at equator)
# Coverage: Global (-180 to 180 lon, -90 to 90 lat)
# Grid: 21,600 x 10,800 pixels
# Values: Surface elevation in meters (negative = bathymetry/ocean depth)
# Range: ~-10,000m (ocean trenches) to ~8,849m (Mt. Everest)
# Use case: High-frequency regression task - tests model ability to capture
#           sharp elevation changes (mountains, valleys, coastlines)
# Note: This is the "ice surface" version (top of ice sheets, not bedrock)
# =============================================================================
echo ""
echo "[2/6] Downloading ETOPO 2022 60-Second Elevation Data..."
echo "      Source: NOAA NCEI"
echo "      Size: ~478 MB"
echo "      Format: NetCDF (global DEM grid)"
echo "      Resolution: 60 arc-seconds (~2 km)"
curl -L -o datasets/elevation/etopo_60s.nc \
    "https://www.ngdc.noaa.gov/thredds/fileServer/global/ETOPO2022/60s/60s_surface_elev_netcdf/ETOPO_2022_v1_60s_N90W180_surface.nc"
echo "      Done: datasets/elevation/etopo_60s.nc"

# =============================================================================
# 3. WWF RESOLVE ECOREGIONS 2017 (~4 MB zipped, ~150 MB unzipped)
# =============================================================================
# Source: World Wildlife Fund / RESOLVE
# Dataset: Terrestrial Ecoregions of the World (TEOW) 2017 update
# Format: Shapefile (.shp + associated files)
# Content: 846 terrestrial ecoregions grouped into 14 biomes and 8 realms
# Attributes:
#   - ECO_NAME: Ecoregion name (846 unique values)
#   - BIOME_NAME/BIOME_NUM: Biome classification (14 classes)
#   - REALM: Biogeographic realm (8 classes: Nearctic, Palearctic, etc.)
#   - NNH: Nature Needs Half conservation status
# Use case: Multi-level classification task - tests hierarchical geographic
#           understanding at different spatial scales
# Citation: Dinerstein et al. (2017) "An Ecoregion-Based Approach to
#           Protecting Half the Terrestrial Realm"
# =============================================================================
echo ""
echo "[3/6] Downloading WWF RESOLVE Ecoregions 2017..."
echo "      Source: Google Cloud Storage (WWF)"
echo "      Size: ~4 MB (zipped)"
echo "      Format: Shapefile (846 ecoregions, 14 biomes, 8 realms)"
curl -L -o datasets/ecoregions/Ecoregions2017.zip \
    "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"
echo "      Extracting..."
unzip -q -o datasets/ecoregions/Ecoregions2017.zip -d datasets/ecoregions/
echo "      Done: datasets/ecoregions/"

# =============================================================================
# 4. NATURAL EARTH DATASETS (~7 MB total)
# =============================================================================
# Source: Natural Earth (naturalearthdata.com) - public domain map data
# Format: Shapefiles (.shp + associated files)
#
# 4a. 10m Coastlines (~3 MB)
#     Resolution: 1:10 million scale
#     Content: 4,133 coastline segments worldwide
#     Use case: Sharp boundary detection - coastline vs. inland classification
#
# 4b. 10m Admin 2 Counties (~3 MB)
#     Resolution: 1:10 million scale
#     Content: US county boundaries and equivalents worldwide
#     Use case: Fine-grained administrative boundary classification
#
# 4c. 110m Countries (~1 MB)
#     Resolution: 1:110 million scale (coarse, for visualization)
#     Content: Country boundaries
#     Use case: Coarse geographic classification, visualization
# =============================================================================
echo ""
echo "[4/6] Downloading Natural Earth Datasets..."
echo "      Source: Natural Earth Data CDN"

echo "      4a. 10m Coastlines (~3 MB)..."
curl -L -o datasets/natural_earth/ne_10m_coastline.zip \
    "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
unzip -q -o datasets/natural_earth/ne_10m_coastline.zip -d datasets/natural_earth/coastline/

echo "      4b. 10m Admin 2 Counties (~3 MB)..."
curl -L -o datasets/natural_earth/ne_10m_admin_2_counties.zip \
    "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_2_counties.zip"
unzip -q -o datasets/natural_earth/ne_10m_admin_2_counties.zip -d datasets/natural_earth/counties/

echo "      4c. 110m Countries (~1 MB)..."
curl -L -o datasets/natural_earth/ne_110m_admin_0_countries.zip \
    "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
unzip -q -o datasets/natural_earth/ne_110m_admin_0_countries.zip -d datasets/natural_earth/countries/

echo "      Done: datasets/natural_earth/"

# =============================================================================
# 5. SATCLIP PRETRAINED MODELS (via Python/Hugging Face)
# =============================================================================
# Source: Microsoft Hugging Face Hub
# Models: SatCLIP - Satellite Contrastive Location-Image Pretraining
#
# Architecture: Location encoder trained via contrastive learning with
#               Sentinel-2 satellite imagery (S2-100K dataset)
#
# Available models:
#   - SatCLIP-ViT16-L10 (103 MB): L=10 spherical harmonics, ViT-16 image encoder
#   - SatCLIP-ViT16-L40 (121 MB): L=40 spherical harmonics, ViT-16 image encoder
#   - SatCLIP-ResNet18-L40: ResNet-18 image encoder variant
#   - SatCLIP-ResNet50-L40: ResNet-50 image encoder variant
#
# L parameter: Maximum degree of spherical harmonic positional encoding
#   - L=10: Coarser resolution (~1000 km), better for large-scale patterns
#   - L=40: Finer resolution (~200 km), better for local variations
#
# Output: 512-dimensional location embeddings
# Use case: Geographic feature extraction, transfer learning for geo tasks
# Citation: Klemmer et al. (2023) "SatCLIP: Global, General-Purpose Location
#           Embeddings with Satellite Imagery"
# =============================================================================
echo ""
echo "[5/6] Downloading SatCLIP Pretrained Models..."
echo "      Source: Microsoft Hugging Face Hub"
echo "      Note: Requires Python with huggingface_hub package"

# Check if Python is available
if command -v python3 &> /dev/null; then
    echo "      Installing huggingface_hub if needed..."
    pip3 install -q huggingface_hub 2>/dev/null || pip install -q huggingface_hub 2>/dev/null || true

    echo "      Downloading SatCLIP-ViT16-L10 (103 MB)..."
    python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('datasets/models', exist_ok=True)
hf_hub_download('microsoft/SatCLIP-ViT16-L10', 'satclip-vit16-l10.ckpt', local_dir='datasets/models')
print('      Downloaded: datasets/models/satclip-vit16-l10.ckpt')
"

    echo "      Downloading SatCLIP-ViT16-L40 (121 MB)..."
    python3 -c "
from huggingface_hub import hf_hub_download
import os
hf_hub_download('microsoft/SatCLIP-ViT16-L40', 'satclip-vit16-l40.ckpt', local_dir='datasets/models')
print('      Downloaded: datasets/models/satclip-vit16-l40.ckpt')
"
    echo "      Done: datasets/models/"
else
    echo "      WARNING: Python not found. Skipping model downloads."
    echo "      To download models manually, run:"
    echo "        pip install huggingface_hub"
    echo "        python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('microsoft/SatCLIP-ViT16-L10', 'satclip-vit16-l10.ckpt', local_dir='datasets/models')\""
    echo "        python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('microsoft/SatCLIP-ViT16-L40', 'satclip-vit16-l40.ckpt', local_dir='datasets/models')\""
fi

# =============================================================================
# 6. GPW v4 POPULATION DENSITY (Manual Download Required)
# =============================================================================
# Source: NASA SEDAC (Socioeconomic Data and Applications Center)
# Dataset: Gridded Population of the World, Version 4, Revision 11
# Format: GeoTIFF (.tif) - georeferenced raster
#
# Available resolutions:
#   - 30 arc-second (~1 km): Highest resolution, largest file (~500 MB)
#   - 2.5 arc-minute (~5 km): Medium resolution
#   - 15 arc-minute (~25 km): Lower resolution
#   - 30 arc-minute (~50 km): Coarse resolution
#
# Values: Population density (persons per square kilometer)
# Years: 2000, 2005, 2010, 2015, 2020
# Coverage: Global land areas
#
# Use case: Smooth regression task - population density varies gradually
#           and tests model ability to capture large-scale spatial patterns
#
# IMPORTANT: Requires free registration at:
#   https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11
#
# After registration, download:
#   gpw-v4-population-density-rev11_2020_30_sec_tif.zip (30-second resolution)
#   OR gpw-v4-population-density-rev11_2020_2pt5_min_tif.zip (2.5-minute)
# =============================================================================
echo ""
echo "[6/6] GPW v4 Population Density Data"
echo "      NOTE: This dataset requires manual download due to registration."
echo ""
echo "      Steps to download:"
echo "      1. Go to: https://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev11"
echo "      2. Create a free NASA Earthdata account (if needed)"
echo "      3. Download the 2020 GeoTIFF at your preferred resolution:"
echo "         - 30 arc-second (~1 km): ~500 MB - highest detail"
echo "         - 2.5 arc-minute (~5 km): ~50 MB - good balance"
echo "         - 15 arc-minute (~25 km): ~5 MB - fast downloads"
echo "      4. Place the extracted .tif file in: datasets/population/"
echo ""

# =============================================================================
# SUMMARY
# =============================================================================
echo ""
echo "============================================================================="
echo "DOWNLOAD COMPLETE"
echo "============================================================================="
echo ""
echo "Directory structure:"
echo "  datasets/"
echo "  ├── temperature/"
echo "  │   └── temperature.csv          (~375 KB)  Global air temperature"
echo "  ├── elevation/"
echo "  │   └── etopo_60s.nc             (~478 MB)  ETOPO 2022 elevation"
echo "  ├── ecoregions/"
echo "  │   └── Ecoregions2017.shp + ... (~150 MB)  WWF ecoregions"
echo "  ├── natural_earth/"
echo "  │   ├── coastline/               (~3 MB)    10m coastlines"
echo "  │   ├── counties/                (~3 MB)    10m admin boundaries"
echo "  │   └── countries/               (~1 MB)    110m countries"
echo "  ├── models/"
echo "  │   ├── satclip-vit16-l10.ckpt   (~103 MB)  SatCLIP L=10 model"
echo "  │   └── satclip-vit16-l40.ckpt   (~121 MB)  SatCLIP L=40 model"
echo "  └── population/                            (Manual download required)"
echo ""
echo "Total downloaded: ~860 MB (excluding population data)"
echo ""
echo "============================================================================="
echo "DATASET DESCRIPTIONS"
echo "============================================================================="
echo ""
echo "1. TEMPERATURE (temperature.csv)"
echo "   - Point observations from weather stations worldwide"
echo "   - Used for: Benchmark regression task"
echo "   - Paper reference: SatCLIP Section 4.1"
echo ""
echo "2. ELEVATION (etopo_60s.nc)"
echo "   - Global digital elevation model from NOAA"
echo "   - Resolution: 60 arc-seconds (~2 km)"
echo "   - Used for: High-frequency regression (sharp terrain features)"
echo ""
echo "3. ECOREGIONS (Ecoregions2017.shp)"
echo "   - 846 terrestrial ecoregions in 14 biomes"
echo "   - Used for: Multi-level classification at different scales"
echo ""
echo "4. NATURAL EARTH (various shapefiles)"
echo "   - Coastlines, counties, countries"
echo "   - Used for: Boundary detection and classification tasks"
echo ""
echo "5. SATCLIP MODELS (.ckpt files)"
echo "   - Pretrained location encoders from Microsoft"
echo "   - L=10: Coarse resolution (~1000 km features)"
echo "   - L=40: Fine resolution (~200 km features)"
echo ""
echo "6. POPULATION (manual download)"
echo "   - Gridded Population of the World v4"
echo "   - Used for: Smooth regression task"
echo "   - Download from: https://sedac.ciesin.columbia.edu"
echo ""
echo "============================================================================="
