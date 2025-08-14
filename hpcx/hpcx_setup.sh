#!/bin/bash
# install hpcx
tar -xvf /opt/hpcx-v2.23-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64.tbz -C /opt; \
mv /opt/hpcx-v2.23-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64 /opt/hpcx; \
rm /opt/hpcx-v2.23-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64.tbz;

# Assuming you have nccl on system 
# check with dpkg -l | grep -i nccl
git clone https://github.com/NVIDIA/nccl-tests.git /home/ubuntu/nccl-tests-hpcx

cd /home/ubuntu/nccl-tests-hpcx

make MPI=1 -j$(nproc)


# Modify based on /etc/hosts for workernodes DNS
cat > hostfile.txt << EOF
hpcx-quest-1
hpcx-quest-2
EOF

mpirun -np 16 -N 8 -x NCCL_NET_PLUGIN=/opt/hpcx/nccl_rdma_sharp_plugin/lib/libnccl-net.so -hostfile hostfile.txt ./build/all_reduce_perf -b 512M -e 8G -f 2 -g 1

