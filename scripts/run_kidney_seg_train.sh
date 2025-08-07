#!/bin/bash
#SBATCH --job-name=train_unet3d
#SBATCH --output=logs/train/train_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100

source ~/.bashrc
conda activate kits23-unet

cd $SLURM_SUBMIT_DIR

mkdir -p logs/train

cd kits23/kits23

python models/train_unet_3d.py


