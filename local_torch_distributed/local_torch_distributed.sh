#!/bin/bash
#SBATCH --job-name=simple_ddp
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=24
#SBATCH --output=/home/ubuntu/SLURM_logs/headnode_logs1/ddp_test_%j.out
#SBATCH --error=/home/ubuntu/SLURM_logs/headnode_logs1/ddp_test_%j.err

# Get master node IP address
HEADNODE_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent hosts "$HEADNODE_HOST" | awk '{ print $1 }' | tail -n1)
MASTER_PORT=29500

export MASTER_ADDR
export MASTER_PORT

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"

# Use torchrun with 8 processes per node, 2 nodes = 16 ranks
srun --output=/home/ubuntu/SLURM_logs/workernodes_logs1/ddp_test_%j_node%N.out --error=/home/ubuntu/SLURM_logs/workernodes_logs1/ddp_test_%j_node%N.err torchrun \
  --nnodes=2 \
  --nproc_per_node=8 \
  --rdzv_id=42 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
  /home/ubuntu/cluster-job-orchestration/local_torch_distributed/local_torch_distributed.py 8
