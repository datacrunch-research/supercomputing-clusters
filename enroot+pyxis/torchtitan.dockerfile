# Usinsg official Pytorch 2.7 + CUDA 12.8 base image
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
#FROM ubuntu:22.04

ARG HF_TOKEN    
ENV HF_TOKEN=${HF_TOKEN}

# makes sure the shell used for subsequent RUN commands is exactly Bash, as located in /bin.
SHELL ["/bin/bash", "-c"]

# Install dependencies
# llamacpp gcc compilation tools
RUN apt-get update && apt-get install -y \
    build-essential \
    fzf \
    ripgrep \
    nvtop \
    sudo \
    kmod \
    wget \
    vim \
    git \
    curl \
    bzip2 \
    ca-certificates \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libssl-dev \
    libibverbs1 \
    ibverbs-utils \
    libmlx5-1 \
    infiniband-diags
    # Cleanup command to remove the apt cache and reduce the image size: # IMPORTANT: Enforces using sudo apt update when entering the container
    #&& rm -rf /var/lib/apt/lists/*

# Cloning the repo
RUN git clone https://github.com/pytorch/torchtitan

# Change to the repo directory using WORKDIR
WORKDIR /workspace/torchtitan

RUN mkdir -p /root/.cache/huggingface

RUN pip install -r requirements.txt

# For CUDA 12.8 on worker nodes
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128.

# Download the tokenizer
RUN python3 scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"      


# docker build -f torchtitan.dockerfile --build-arg HF_TOKEN="$HF_TOKEN" -t torchtitan_cuda128_torch27 .
# docker run --gpus all --shm-size 32g --network=host -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface --name torchtitan_workload -it --rm --ipc=host torchtitan_cuda128_torch27 bash -c 'CONFIG_FILE="./train_configs/llama3_8b.toml" ./run_llama_train.sh'