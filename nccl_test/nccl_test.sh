#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpus
#SBATCH --output=SLURM_logs/headnode_logs/nccl_test_%j_headnode.out
#SBATCH --error=SLURM_logs/headnode_logs/nccl_test_%j_headnode.err
#SBATCH --time=00:10:00

# Make sure logs directory exists
mkdir -p SLURM_logs

# Path to the NCCL test binary
NCCL_TEST_BINARY=./nccl-tests/build/all_reduce_perf

# Set up the correct OpenMPI runtime
export PATH=/usr/mpi/gcc/openmpi-4.1.7rc1/bin:$PATH
export LD_LIBRARY_PATH=/usr/mpi/gcc/openmpi-4.1.7rc1/lib:$LD_LIBRARY_PATH


# Show nodes involved
echo "[INFO] Running on nodes:"
scontrol show hostnames $SLURM_NODELIST

# Run the test
srun --mpi=pmix --output=SLURM_logs/workernodes_logs1/nccl_test_%j_node%N.out --error=SLURM_logs/workernodes_logs1/nccl_test_%j_node%N.err $NCCL_TEST_BINARY -b 1M -e 1G -f 2 -g 8
