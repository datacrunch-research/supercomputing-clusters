#TODO: Not working




#!/bin/bash

# Azure-compatible NCCL All-Reduce Test
# This script replicates Azure HPC Health Checks NCCL testing methodology

# Set Azure's NCCL environment variables for single-node NVLink testing
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
export NCCL_IB_PCI_RELAXED_ORDERING=1
export UCX_IB_PCI_RELAXED_ORDERING=on
export UCX_TLS=tcp
export UCX_NET_DEVICES=eth0
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_SOCKET_IFNAME=eth0

# Single-node optimizations (no IB, pure NVLink)
export NCCL_SHM_DISABLE=0     # Enable shared memory for single-node
export NCCL_P2P_DISABLE=0     # Enable NVLink P2P for single-node
export NCCL_IB_DISABLE=1      # Disable InfiniBand for single-node testing
export NCCL_NET_DISABLE=1     # Disable network for single-node
export NCCL_DEBUG=INFO

# Optional: Additional optimization variables
# export NCCL_ALGO=Ring
# export NCCL_MIN_NCHANNELS=16
# export NCCL_MAX_NCHANNELS=16
# export NCCL_BUFFSIZE=8388608

# Configuration
GPU_COUNT=${1:-8}           # Number of GPUs to test
MESSAGE_SIZE=${2:-"16G"}    # Message size (e.g., 1G, 16G, 32G)
ITERATIONS=${3:-1}          # Number of iterations per test
NCCL_TESTS_PATH=${4:-"/home/rodri/nccl-tests/build/all_reduce_perf"}

echo "=== Azure-Style NCCL All-Reduce Test ==="
echo "GPU Count: $GPU_COUNT"
echo "Message Size: $MESSAGE_SIZE"
echo "Iterations: $ITERATIONS"
echo "NCCL Tests Path: $NCCL_TESTS_PATH"
echo

# Check if nccl-tests exists
if [ ! -f "$NCCL_TESTS_PATH" ]; then
    echo "NCCL tests not found at: $NCCL_TESTS_PATH"
    echo "   Install with: git clone https://github.com/NVIDIA/nccl-tests.git && cd nccl-tests && make"
    exit 1
fi

# Azure's MPI configuration
MPI_ARGS="-np $GPU_COUNT --map-by ppr:1:gpu -bind-to gpu -mca coll_hcoll_enable 0 --allow-run-as-root"

# Azure's NCCL test arguments
NCCL_ARGS="-b $MESSAGE_SIZE -f 2 -g 1 -e $MESSAGE_SIZE -c $ITERATIONS"

echo "=== Running Azure-style NCCL test ==="
echo "Command: mpirun $MPI_ARGS $NCCL_TESTS_PATH $NCCL_ARGS"
echo

# Run the test
mpirun $MPI_ARGS $NCCL_TESTS_PATH $NCCL_ARGS

# Check exit code
if [ $? -eq 0 ]; then
    echo "NCCL test completed successfully"
else
    echo "NCCL test failed"
    exit 1
fi 