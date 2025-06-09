#!/bin/bash
sudo git clone https://github.com/pytorch/torchtitan /home/ubuntu/torchtitan
sudo chown -R ubuntu:ubuntu /home/ubuntu/torchtitan
cd /home/ubuntu/torchtitan
pip install -r requirements.txt

echo export PATH=/home/ubuntu/.local/bin:$PATH >> /home/ubuntu/.bashrc
source /home/ubuntu/.bashrc

huggingface-cli login

# For CUDA 12.8 on worker nodes
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128.

python3 scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"      