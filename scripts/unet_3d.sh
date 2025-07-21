#!/bin/bash
#SBATCH --job-name=train_unet3d
#SBATCH --output=/net/tscratch/people/plgmpniak/KITS_project/Kits_23/kits23/logs/train/train_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100

# Aktywacja środowiska
source ~/.bashrc
conda activate kits23-unet

# Przejście do katalogu projektu
cd $SLURM_SUBMIT_DIR

# Tworzenie katalogu na logi
mkdir -p logs/train

cd kits23/kits23

# Uruchom trening
python models/train_unet_3d.py


