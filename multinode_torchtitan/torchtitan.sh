#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=50
#SBATCH --partition=gpus
#SBATCH --job-name=torchtitan
#SBATCH -o /home/ubuntu/slurm_logging/headnode/%x_%j_headnode.out
#SBATCH -e /home/ubuntu/slurm_logging/headnode/%x_%j_headnode.err

# =======================
# Safe MASTER_ADDR setup
# =======================
HEADNODE_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent hosts "$HEADNODE_HOST" | grep -Eo '10\.[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
MASTER_PORT=$((5000 + SLURM_JOB_ID % 10000))  # fallback using modulo to stay within safe port range

export MASTER_ADDR
export MASTER_PORT

echo "======== Distributed Config ========"
echo "HEADNODE_HOST: $HEADNODE_HOST"
echo "Resolved MASTER_ADDR: $MASTER_ADDR"
echo "Assigned MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "All Hosts:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "===================================="



export TORCHELASTIC_ERROR_FILE=./error-${SLURM_JOBID}-${SLURM_NODEID}.json
export OMP_NUM_THREADS=1
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/torchtitan

export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export CUDA_LAUNCH_BLOCKING=0

# on your cluster you might need these:
# set the network interface
export NCCL_BUFFSIZE=2097152
#export TORCH_DIST_INIT_BARRIER=1
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0
export NCCL_IB_HCA=mlx5
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_NET_GDR_LEVEL=5
export NCCL_IB_DISABLE=0
export NCCL_NET=IB


#printenv

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}

# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun --output=/home/ubuntu/slurm_logging/workernodes/%x_%j_node%N.out --error=/home/ubuntu/slurm_logging/workernodes/%x_%j_node%N.err torchrun \
    --nnodes=2 \
    --nproc_per_node=8 \
    --rdzv_id=101 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    ./torchtitan/train.py --job.config_file ${CONFIG_FILE}


# sbatch torchtitan_multinode.sh
