1. Clone nccl-test official repo:
```bash
sudo git clone https://github.com/NVIDIA/nccl-tests.git
cd nccl-tests/
# you need mpi bin location if custom
sudo make MPI=1 MPI_HOME=/usr/mpi/gcc/openmpi-4.1.7rc1 CUDA_HOME=/usr/local/cuda
```

2. Slurm script:
- Specify output and error logs location
- Specify nccl bin location
- Specify LD_LIBRARY_PATH
```bash
#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpus
#SBATCH --output=SLURM_logs/nccl_test_%j_headnode.out
#SBATCH --error=SLURM_logs/nccl_test_%j_headnode.err
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
srun --mpi=pmix --output=SLURM_logs/nccl_test_%j_node%N.out --error=SLURM_logs/nccl_test_%j_node%N.err $NCCL_TEST_BINARY -b 1M -e 1G -f 2 -g 8
```

Output examople for workernode1.out:

```bash
# nThread 1 nGpus 8 minBytes 1048576 maxBytes 1073741824 step: 2(factor) warmup iters: 5 iters: 20 agg iters: 1 validation: 1 graph: 0
#
# Using devices
#  Rank  0 Group  0 Pid   7012 on on-demand-testing-rodri4-1 device  0 [0000:68:00] NVIDIA H200
#  Rank  1 Group  0 Pid   7012 on on-demand-testing-rodri4-1 device  1 [0000:6c:00] NVIDIA H200
#  Rank  2 Group  0 Pid   7012 on on-demand-testing-rodri4-1 device  2 [0000:72:00] NVIDIA H200
#  Rank  3 Group  0 Pid   7012 on on-demand-testing-rodri4-1 device  3 [0000:76:00] NVIDIA H200
#  Rank  4 Group  0 Pid   7012 on on-demand-testing-rodri4-1 device  4 [0000:cc:00] NVIDIA H200
#  Rank  5 Group  0 Pid   7012 on on-demand-testing-rodri4-1 device  5 [0000:d0:00] NVIDIA H200
#  Rank  6 Group  0 Pid   7012 on on-demand-testing-rodri4-1 device  6 [0000:d5:00] NVIDIA H200
#  Rank  7 Group  0 Pid   7012 on on-demand-testing-rodri4-1 device  7 [0000:db:00] NVIDIA H200
#  Rank  8 Group  0 Pid   6794 on on-demand-testing-rodri4-2 device  0 [0000:68:00] NVIDIA H200
#  Rank  9 Group  0 Pid   6794 on on-demand-testing-rodri4-2 device  1 [0000:6c:00] NVIDIA H200
#  Rank 10 Group  0 Pid   6794 on on-demand-testing-rodri4-2 device  2 [0000:72:00] NVIDIA H200
#  Rank 11 Group  0 Pid   6794 on on-demand-testing-rodri4-2 device  3 [0000:76:00] NVIDIA H200
#  Rank 12 Group  0 Pid   6794 on on-demand-testing-rodri4-2 device  4 [0000:cc:00] NVIDIA H200
#  Rank 13 Group  0 Pid   6794 on on-demand-testing-rodri4-2 device  5 [0000:d0:00] NVIDIA H200
#  Rank 14 Group  0 Pid   6794 on on-demand-testing-rodri4-2 device  6 [0000:d5:00] NVIDIA H200
#  Rank 15 Group  0 Pid   6794 on on-demand-testing-rodri4-2 device  7 [0000:db:00] NVIDIA H200
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)       
     1048576        262144     float     sum      -1    258.6    4.05    7.60      0    268.3    3.91    7.33      0
     2097152        524288     float     sum      -1    288.9    7.26   13.61      0    513.0    4.09    7.66      0
     4194304       1048576     float     sum      -1    687.0    6.11   11.45      0    368.2   11.39   21.36      0
     8388608       2097152     float     sum      -1    379.6   22.10   41.43      0   1196.7    7.01   13.14      0
    16777216       4194304     float     sum      -1    521.9   32.15   60.27      0    514.5   32.61   61.14      0
    33554432       8388608     float     sum      -1    744.8   45.05   84.47      0    880.4   38.11   71.46      0
    67108864      16777216     float     sum      -1   1128.2   59.48  111.53      0    890.8   75.34  141.26      0
   134217728      33554432     float     sum      -1   1368.1   98.11  183.95      0   1343.6   99.89  187.29      0
   268435456      67108864     float     sum      -1   2476.2  108.40  203.26      0   2597.1  103.36  193.80      0
   536870912     134217728     float     sum      -1   4356.2  123.24  231.08      0   4165.0  128.90  241.69      0
  1073741824     268435456     float     sum      -1   7518.6  142.81  267.77      0   7743.4  138.67  260.00      0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 110.117 
#

```