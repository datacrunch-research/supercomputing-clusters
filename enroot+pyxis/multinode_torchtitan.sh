#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=50
#SBATCH --partition=gpus
#SBATCH --job-name=torchtitan_multinode
#SBATCH -o /home/ubuntu/slurm_logging/headnode/%x_%j_headnode.out
#SBATCH -e /home/ubuntu/slurm_logging/headnode/%x_%j_headnode.err

# === Compute these HOST-side ===
HEADNODE_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent hosts "$HEADNODE_HOST" | grep -Eo '10\.[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
MASTER_PORT=$((5000 + SLURM_JOB_ID % 10000))

CONFIG_FILE=${CONFIG_FILE:-"torchtitan/models/llama3/train_configs/llama3_70b.toml"}


echo "======== Distributed Config ========"
echo "HEADNODE_HOST: $HEADNODE_HOST"
echo "Resolved MASTER_ADDR: $MASTER_ADDR"
echo "Assigned MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "All Hosts:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "===================================="


# === Launch the container job ===
srun \
  --container-image=/home/ubuntu/torchtitan_cuda128_torch27.sqsh \
  --container-mounts=/home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  --container-writable \
  --export=ALL,HEADNODE_HOST=$HEADNODE_HOST,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT,NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=ALL \
  torchrun --nnodes=2 --nproc_per_node=8 \
           --rdzv_backend=c10d \
           --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
           -m torchtitan.train --job.config_file ${CONFIG_FILE}
