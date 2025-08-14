#!/bin/bash

# Assuming you have nccl on system and lmod
# check with dpkg -l | grep -i nccl

module load hpcx
sudo git clone https://github.com/NVIDIA/nccl-tests.git /home/ubuntu/nccl-tests-hpcx
sudo chown -R ubuntu:ubuntu /home/ubuntu/nccl-tests-hpcx

cd /home/ubuntu/nccl-tests-hpcx

make MPI=1 -j$(nproc)


# Modify based on /etc/hosts for workernodes DNS
cat > hostfile.txt << EOF
brave-book-blows-fin-03-1
brave-book-blows-fin-03-2
EOF



mpirun -np 16 -N 8 -x NCCL_NET_PLUGIN=/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so -hostfile hostfile.txt ./build/all_reduce_perf -b 512M -e 8G -f 2 -g 1

module unload hpcx