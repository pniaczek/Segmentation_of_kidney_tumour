#!/bin/bash
#SBATCH --job-name=crop_kits23_128
#SBATCH --output=logs/crop/crop_128_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100
#SBATCH --gres=gpu:1

source ~/.bashrc

# Use all 8 CPUs for NumPy/BLAS (and avoid oversubscription)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1

# Env
source ~/.bashrc
conda activate kits23-augment

# Logs
mkdir -p logs/crop

# Go to project
cd /net/tscratch/people/plgmpniak/KITS_project/Kits_23

# Run the patch maker:
# - images: clip [-80,310] + z-score using global mean/std from first 5 images
# - labels: not normalized
# - 128x128x128 patches
# - 1 train + 1 test patch per case
python /net/tscratch/people/plgmpniak/KITS_project/Kits_23/preprocessing/make_cropped_patches.py \
  --input_dir  /net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/augumented_dataset \
  --output_dir /net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/cropped_smaller_dataset_z \
  --patch 128 128 128 \
  --patches_per_pair_train 1 \
  --patches_per_pair_test  1
