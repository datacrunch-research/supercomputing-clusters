#!/bin/bash
# install hpcx (needs sudo access) (need to be installed on all nodes)
# download hpcx from https://developer.nvidia.com/networking/hpc-x
tar -xvf hpcx-v2.23-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64.tbz -C /opt; \
mv /opt/hpcx-v2.23-gcc-doca_ofed-ubuntu22.04-cuda12-x86_64 /opt/hpcx;

echo 'export HPCX_HOME=/opt/hpcx' >> ~/.bashrc
echo 'PATH=$HPCX_HOME/ompi/bin:$PATH' >> ~/.bashrc


#install lmod
sudo apt-get install -y lmod
echo 'source /etc/profile.d/lmod.sh' >> ~/.bashrc

# Testing
# module use $HPCX_HOME/modulefiles
# module load hpcx
# env | grep HPCX