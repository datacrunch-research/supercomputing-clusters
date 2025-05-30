#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=50
#SBATCH --partition=gpus
#SBATCH --job-name=torchtitan_singlenode
#SBATCH -o /home/ubuntu/slurm_logging/headnode/%x_%j_headnode.out
#SBATCH -e /home/ubuntu/slurm_logging/headnode/%x_%j_headnode.err
#SBATCH --container-image=/home/ubuntu/torchtitan_cuda128_torch27.sqsh
#SBATCH --container-name=torchtitan_singlenode
#SBATCH --container-mounts=/home/ubuntu/.cache/huggingface:/root/.cache/huggingface
#SBATCH --no-container-mount-home
#SBATCH --container-remap-root
#SBATCH --container-writable

CONFIG_FILE=${CONFIG_FILE:-"torchtitan/models/llama3/train_configs/debug_model.toml"}

# ETA: 10 minutes
bash run_train.sh --job.config_file ${CONFIG_FILE}