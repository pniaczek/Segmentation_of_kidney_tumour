#!/bin/bash
#SBATCH --job-name=visualize_kits23
#SBATCH --output=logs/visualizations/visualize_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100

source ~/.bashrc
conda activate kits23-unet

cd $SLURM_SUBMIT_DIR

mkdir -p logs/visualizations

python -m kits23.visualizations.visualization preprocessed kits23/visualizations/outputs
