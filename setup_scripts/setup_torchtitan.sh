#!/bin/bash

# Check if a username argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <username>"
    exit 1
fi

# Assign the first argument to USERNAME
USERNAME=$1

# Replace 'ubuntu' with the provided username
git clone https://github.com/pytorch/torchtitan /home/$USERNAME/torchtitan
cd /home/$USERNAME/torchtitan
pip install -r requirements.txt

echo export PATH=/home/$USERNAME/.local/bin:$PATH >> /home/$USERNAME/.bashrc
source /home/$USERNAME/.bashrc

huggingface-cli login

# For CUDA 12.8 on worker nodes
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128.
# For B200 support
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

python3 scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"   

# Usage: bash setup_torchtitan.sh rodri