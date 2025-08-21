#!/bin/bash
#SBATCH --job-name=train_kits23
#SBATCH --output=logs/train/resnet_train_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100

source ~/.bashrc
conda activate kits23-unet

cd $SLURM_SUBMIT_DIR
cd /net/tscratch/people/plgmpniak/KITS_project/Kits_23

python -m train.train_resUnet
