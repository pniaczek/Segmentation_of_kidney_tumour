#!/bin/bash
#SBATCH --job-name=check_cropping
#SBATCH --output=logs/preprocess/check_cropping_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100

set -euo pipefail

# Env
source ~/.bashrc
conda activate kits23-unet

# Ensure log dir exists
mkdir -p logs/preprocess

# Go to project root (optional)
cd /net/tscratch/people/plgmpniak/KITS_project/Kits_23

# Run the check script (forwards any extra args passed to sbatch)
python /net/tscratch/people/plgmpniak/KITS_project/Kits_23/preprocessing/check_cropping.py "$@"
