# Task for HPC Claude: Start S2K Dataset Preprocessing

## Context

The user needs to preprocess the S2K (SatCLIP) Sentinel-2 multispectral dataset. This is a compute-intensive task that takes 4-8 hours. The preprocessing pipeline:
1. Extracts tar.xz archives containing satellite imagery
2. Converts TIF files to HuggingFace Arrow format for fast loading
3. Normalizes pixel values and pads from 12 to 13 channels

## Your Task

1. **Pull the latest code** (if not already done):
   ```bash
   cd /path/to/learned_activation/satclip
   git pull origin main
   ```

2. **Verify the data directory exists** and contains the raw archives:
   ```bash
   ls -la /projects/bdbk/cherd/data/satclip_manual/archives/
   ```
   You should see files like `s2k_all_*.tar.xz`

3. **Update paths if necessary**:
   - If the data is in a different location, edit:
     - `experiments/scripts/slurm/preprocess_s2k.sh` (lines 48-52)
     - `experiments/configs/experiments/contrastive_multispectral.yaml` (line 72)

4. **Create the logs directory**:
   ```bash
   mkdir -p logs
   ```

5. **Submit the preprocessing job**:
   ```bash
   sbatch experiments/scripts/slurm/preprocess_s2k.sh
   ```

6. **Verify the job was submitted**:
   ```bash
   squeue -u $USER
   ```
   You should see a job named `preprocess_s2k` in the queue.

7. **Monitor initial progress** (optional):
   ```bash
   # Wait a minute for job to start, then:
   tail -f logs/preprocess_s2k_*.out
   ```

## Expected Output

The preprocessing will create:
- `/projects/bdbk/cherd/data/satclip_manual/raw_tifs/` - Extracted TIF files
- `/projects/bdbk/cherd/data/satclip_manual/satclip_hf/` - HuggingFace Arrow dataset
- `/projects/bdbk/cherd/data/satclip_manual/satclip_hf_preprocessed/` - Normalized dataset

## After Preprocessing Completes

Run the HPC test suite to verify everything works:
```bash
python experiments/scripts/hpc/test_hpc.py --data-dir /projects/bdbk/cherd/data/satclip_manual
```

## Troubleshooting

- **Job fails immediately**: Check `logs/preprocess_s2k_*.err` for errors
- **Out of disk space**: Check with `df -h` and clean up if needed
- **Archives not found**: Verify the path to tar.xz files in the SLURM script

## SLURM Job Details

The preprocessing job requests:
- 12 hours wall time
- 64GB RAM
- 16 CPUs
- No GPU (CPU-only task)
- Partition: condo-jacobsn

If you need to modify these, edit `experiments/scripts/slurm/preprocess_s2k.sh`.
