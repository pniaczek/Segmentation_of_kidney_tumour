#!/bin/bash
#SBATCH --job-name=preprocess_kits23
#SBATCH --output=logs/preprocess/preprocess_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100

source ~/.bashrc
conda activate kits23-unet

cd $SLURM_SUBMIT_DIR

mkdir -p logs/preprocess

python /net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/kits23/models/kidney_segmentation.py
