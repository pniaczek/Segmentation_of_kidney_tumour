#!/bin/bash
#SBATCH --job-name=train_kits23_3cls
#SBATCH --output=logs/train/train_3cls_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12           # allocate more CPU threads (4 per GPU)
#SBATCH --mem=128G
#SBATCH --gres=gpu:3                 # request 3 GPUs
#SBATCH --time=06:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --account=plglscclass24-gpu-a100


# --- Env ---
source ~/.bashrc
conda activate kits23-unet

# Threads
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}

# NCCL setup (good practice for multi-GPU)
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1

# --- Paths ---
PROJECT_DIR="/net/tscratch/people/plgmpniak/KITS_project/Kits_23"
TRAIN_SCRIPT="${PROJECT_DIR}/train/train_3_classes.py"
TRAIN_DATA="${PROJECT_DIR}/data/cropped_smaller_dataset_z/train"
VAL_DATA="${PROJECT_DIR}/data/cropped_smaller_dataset_z/test"

mkdir -p logs/train
cd "${PROJECT_DIR}"

# --- Training args ---
EPOCHS=50
BS=2                                 # per-GPU batch size (so effective = 3×BS)
WORKERS=4
VAL_EVERY=2
SAVE_PATH="trained_models/three_classes"

ARGS=(
  --train_data_path "${TRAIN_DATA}"
  --val_data_path   "${VAL_DATA}"
  --epochs ${EPOCHS}
  --batch_size ${BS}
  --num_workers ${WORKERS}
  --val_every ${VAL_EVERY}
  --save_every ${VAL_EVERY}
  --save_path "${SAVE_PATH}"
  --amp_dtype bf16
)

# --- Launch on 3 GPUs with torchrun ---
echo "[INFO] Running DDP on 3 GPUs"
torchrun --nproc_per_node=3 "${TRAIN_SCRIPT}" "${ARGS[@]}"
