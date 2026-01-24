# Task for HPC Claude: S2K Dataset Extraction and Preprocessing

## Overview

This task prepares the SatCLIP/S2K Sentinel-2 multispectral dataset for training. The raw data consists of tar.xz archives containing 12-band GeoTIFF images with geographic coordinates.

## Data Locations

**Raw data (tar archives):**
```
/engrfs/project/jacobsn/hiqbal/data_raw/satclip/images/data_1.tar.xz
/engrfs/project/jacobsn/hiqbal/data_raw/satclip/images/data_2.tar.xz
...
/engrfs/project/jacobsn/hiqbal/data_raw/satclip/images/data_10.tar.xz
```

**Output directory (create if needed):**
```
/engrfs/project/jacobsn/hiqbal/data_processed/satclip/
```

## Prerequisites Check

### 1. Check if index.csv exists

The preprocessing requires an `index.csv` file that maps each .tif filename to its coordinates:

```bash
# Check if index.csv exists in the data directory
ls -la /engrfs/project/jacobsn/hiqbal/data_raw/satclip/index.csv

# If it exists, check the format:
head -5 /engrfs/project/jacobsn/hiqbal/data_raw/satclip/index.csv
```

Expected format:
```csv
fn,lon,lat
1.tif,-122.4194,37.7749
2.tif,-74.0060,40.7128
...
```

**If index.csv doesn't exist:** You need to create it or find where the coordinate metadata is stored. Check for:
- `metadata.csv` or similar files
- JSON files with coordinates
- The original SatCLIP dataset documentation

### 2. Check available disk space

```bash
df -h /engrfs/project/jacobsn/hiqbal/
```

Estimated space needed:
- Extracted TIFs: ~100-200 GB
- HuggingFace dataset: ~100-150 GB
- Preprocessed dataset: ~100-150 GB
- **Total: ~400-500 GB**

## Step-by-Step Instructions

### Step 1: Pull latest code

```bash
cd /path/to/learned_activation/satclip
git pull origin main
```

### Step 2: Create output directories

```bash
OUTPUT_BASE="/engrfs/project/jacobsn/hiqbal/data_processed/satclip"
mkdir -p "${OUTPUT_BASE}/tif"
mkdir -p "${OUTPUT_BASE}/satclip_hf"
mkdir -p "${OUTPUT_BASE}/satclip_hf_preprocessed"
mkdir -p logs
```

### Step 3: Update the SLURM script paths

Edit `experiments/scripts/slurm/preprocess_s2k.sh`:

```bash
# Change these lines (around line 59-64):
DATA_DIR="/engrfs/project/jacobsn/hiqbal/data_raw/satclip"
TIF_DIR="/engrfs/project/jacobsn/hiqbal/data_processed/satclip/tif"
HF_OUTPUT="/engrfs/project/jacobsn/hiqbal/data_processed/satclip/satclip_hf"
PREPROCESSED_OUTPUT="/engrfs/project/jacobsn/hiqbal/data_processed/satclip/satclip_hf_preprocessed"
```

### Step 4: Submit the preprocessing job

```bash
cd /path/to/learned_activation/satclip
sbatch experiments/scripts/slurm/preprocess_s2k.sh
```

### Step 5: Monitor progress

```bash
# Check job status
squeue -u $USER

# View output (once job starts)
tail -f logs/preprocess_s2k_*.out
```

## Alternative: Manual Step-by-Step Execution

If you prefer to run steps manually (useful for debugging):

### Manual Step 1: Extract tar files

```bash
# Activate environment
conda activate satclip

# Set paths
DATA_DIR="/engrfs/project/jacobsn/hiqbal/data_raw/satclip"
TIF_DIR="/engrfs/project/jacobsn/hiqbal/data_processed/satclip/tif"

# Extract (takes 1-2 hours)
python experiments/scripts/preprocessing/build_hf_dataset.py \
    --data-dir "${DATA_DIR}" \
    --tif-dir "${TIF_DIR}" \
    --workers 8 \
    --extract
```

### Manual Step 2: Build HuggingFace dataset

```bash
HF_OUTPUT="/engrfs/project/jacobsn/hiqbal/data_processed/satclip/satclip_hf"

python experiments/scripts/preprocessing/build_hf_dataset.py \
    --data-dir "${DATA_DIR}" \
    --tif-dir "${TIF_DIR}" \
    --output "${HF_OUTPUT}" \
    --workers 8 \
    --build
```

### Manual Step 3: Preprocess (normalize + pad channels)

```bash
PREPROCESSED_OUTPUT="/engrfs/project/jacobsn/hiqbal/data_processed/satclip/satclip_hf_preprocessed"

python experiments/scripts/preprocessing/preprocess_hf_dataset.py \
    --input "${HF_OUTPUT}" \
    --output "${PREPROCESSED_OUTPUT}" \
    --num-proc 8
```

## Expected Output

After preprocessing completes:

```
/engrfs/project/jacobsn/hiqbal/data_processed/satclip/
├── tif/                          # Extracted .tif files (~100K files)
│   ├── 1.tif
│   ├── 2.tif
│   └── ...
├── satclip_hf/                   # HuggingFace Arrow dataset (raw)
│   ├── data-00000-of-00001.arrow
│   ├── dataset_info.json
│   └── state.json
└── satclip_hf_preprocessed/      # Final dataset (normalized, 13 channels)
    ├── data-00000-of-00001.arrow
    ├── dataset_info.json
    └── state.json
```

## Verification

After preprocessing, verify the output:

```python
from datasets import load_from_disk

ds = load_from_disk("/engrfs/project/jacobsn/hiqbal/data_processed/satclip/satclip_hf_preprocessed")
print(f"Dataset size: {len(ds)} samples")
print(f"Sample keys: {ds[0].keys()}")
print(f"Image shape: {ds[0]['image'].shape}")  # Should be (13, H, W)
print(f"Image range: [{ds[0]['image'].min():.3f}, {ds[0]['image'].max():.3f}]")  # Should be ~0-1
```

## Update Training Config

Once preprocessing is complete, update the training config to point to the new dataset:

Edit `experiments/configs/experiments/contrastive_multispectral.yaml`:

```yaml
data:
  hf_dataset_path: "/engrfs/project/jacobsn/hiqbal/data_processed/satclip/satclip_hf_preprocessed"
  pad_to_13_channels: false  # Already padded
  preprocessed: true          # Already normalized
```

## Troubleshooting

### "index.csv not found"
The build script needs `index.csv` to know the coordinates for each image. Check where this file is located:
```bash
find /engrfs/project/jacobsn/hiqbal/data_raw/satclip -name "*.csv" -o -name "*.json"
```

### "No tar.xz files found"
The script looks for `images/data_*.tar.xz` or `data_*.tar.xz`. If your structure is different:
```bash
find /engrfs/project/jacobsn/hiqbal/data_raw/satclip -name "*.tar.xz"
```
Then adjust the paths in the script accordingly.

### Out of memory
Reduce `--workers` or `--num-proc` to a lower number (e.g., 4).

### Disk space issues
The preprocessing generates large intermediate files. Consider:
1. Removing TIF files after building HuggingFace dataset
2. Removing raw HF dataset after preprocessing
