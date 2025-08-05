#!/bin/bash
#SBATCH --job-name=multinode_torch_distributed
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpus
#SBATCH --output=/home/ubuntu/slurm_logging/headnode/multinode_torch_test_%j.out
#SBATCH --error=/home/ubuntu/slurm_logging/headnode/multinode_torch_test_%j.err

# Get master node IP address
HEADNODE_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent hosts "$HEADNODE_HOST" | awk '{ print $1 }' | tail -n1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Use torchrun with 8 processes per node, 2 nodes = 16 ranks
srun --output=/home/ubuntu/slurm_logging/workernodes/multinode_torch_test_%j_node%N.out --error=/home/ubuntu/slurm_logging/workernodes/multinode_torch_test_%j_node%N.err torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_id=42 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  /home/ubuntu/supercomputing-clusters/multinode_torch_distributed/multinode_torch_distributed.py
