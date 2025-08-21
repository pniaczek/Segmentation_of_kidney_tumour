#!/bin/bash
#SBATCH --job-name=train_kits23_3cls_patches
#SBATCH --output=logs/train/train_3cls_patches_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=07:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100

source ~/.bashrc
conda activate kits23-unet

cd /net/tscratch/people/plgmpniak/KITS_project/Kits_23

python train/train_3_classes.py \
  --dataset_type nifti \
  --train_data_path /net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/cropped_smaller_dataset/train \
  --val_data_path   /net/tscratch/people/plgmpniak/KITS_project/Kits_23/data/cropped_smaller_dataset/test \
  --epochs 20 \
  --batch_size 1 \
  --num_workers 1 \
  --val_every 2
