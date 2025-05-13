#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=50
#SBATCH --partition=gpus
#SBATCH --job-name=torchtitan
#SBATCH -o /home/ubuntu/slurm_logging/headnode/%x_%j_headnode.out
#SBATCH -e /home/ubuntu/slurm_logging/headnode/%x_%j_headnode.err
#SBATCH --container-image=/home/ubuntu/torchtitan_cuda128_torch27.sqsh
#SBATCH --container-name=torchtitan
#SBATCH --container-mounts=/home/ubuntu/.cache/huggingface:/root/.cache/huggingface
#SBATCH --no-container-mount-home


# ETA: 10 minutes
echo "Running inside container:"
grep PRETTY /etc/os-release
