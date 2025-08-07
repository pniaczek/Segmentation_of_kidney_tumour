#!/bin/bash
#SBATCH --job-name=augumentation_kits23
#SBATCH --output=logs/preprocess/preprocess_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100

source ~/.bashrc
conda activate kits23-augment

cd $SLURM_SUBMIT_DIR

mkdir -p logs/preprocess

python /net/tscratch/people/plgmpniak/KITS_project/Kits_23/preprocessing/data_augumentation.py \
  --src /net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/cropped_dataset \
  --dst /net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/augumented_dataset \
  --repeats 5 \
  --seed 0