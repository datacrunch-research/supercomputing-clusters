#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=50
#SBATCH --partition=gpus
#SBATCH --job-name=torchtitan_singlenode
#SBATCH -o /home/ubuntu/slurm_logging/headnode/%x_%j.out
#SBATCH -e /home/ubuntu/slurm_logging/headnode/%x_%j.err


CONFIG_FILE=${CONFIG_FILE:-"torchtitan/models/llama3/train_configs/debug_model.toml"}

# ETA: 10 minutes
srun --container-image=/home/ubuntu/torchtitan_cuda128_torch27.sqsh \
    --container-name=torchtitan_singlenode \
    --container-mounts=/home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
   --container-writable \
   --no-container-mount-home \
   bash run_train.sh --job.config_file ${CONFIG_FILE}