## Enroot Configuration

Enroot configuration resides in `/etc/enroot/enroot.conf`. Default config points to:

```bash
ENROOT_LIBRARY_PATH       /usr/lib/enroot             # Path to library sources
ENROOT_SYSCONF_PATH       /etc/enroot                 # Path to system configuration files
ENROOT_RUNTIME_PATH       ${XDG_RUNTIME_DIR}/enroot   # Path to the runtime working directory
ENROOT_CONFIG_PATH        ${XDG_CONFIG_HOME}/enroot   # Path to user configuration files
ENROOT_CACHE_PATH         ${XDG_CACHE_HOME}/enroot    # Path to user image/credentials cache
ENROOT_DATA_PATH          ${XDG_DATA_HOME}/enroot     # Path to user container storage
ENROOT_TEMP_PATH          ${TMPDIR}                   # Path to temporary directory
```

XDG variables point to `/home/*user*`, thus the NFS. As we would like to avoid concurrent enroot file conflicts and enable a multiuser, multijobs setup, we have to set per-node local directories in `/etc/enroot/enroot.conf`:

```bash
ENROOT_LIBRARY_PATH        /usr/lib/enroot
ENROOT_SYSCONF_PATH        /etc/enroot
ENROOT_RUNTIME_PATH        /mnt/local_disk/enroot/runtime/$UID
ENROOT_CONFIG_PATH         /mnt/local_disk/enroot/config
ENROOT_CACHE_PATH          /mnt/local_disk/enroot/cache/$UID
ENROOT_DATA_PATH           /mnt/local_disk/enroot/data/$UID
ENROOT_TEMP_PATH           ${TMPDIR:-/tmp}
```

- **Issue:** `slurmstepd: error: pyxis:     mkdir: cannot create directory ‘/mnt/local_disk/enroot/cache/.tokens.1000’: Permission denied` this is due to permissions issues from different users writes. The problem resides in slurm having trouble transferring linux groups to slurm scripts environment.

  **Solution:** Make enroot folder world-writable

  Make the enroot directory world-writable so jobs can write there regardless of group:

```bash
sudo chmod -R a+rwx /mnt/local_disk/enroot
```

​	[mkdir permission denied pyxis ($uid)](https://github.com/NVIDIA/pyxis/issues/138)

- **Issue**: `[ERROR] /etc/enroot/hooks.d/98-nvidia.sh exited with return code 1`

  ```bash
  slurmstepd: error: pyxis: container start failed with error code: 1
  slurmstepd: error: pyxis: printing enroot log file:
  slurmstepd: error: pyxis:     nvidia-container-cli: container error: file lookup failed: /proc/55900/root/mnt/local_disk/enroot/data/pyxis_torchtitan_singlenode/etc/debian_version: permission denied
  slurmstepd: error: pyxis:     [ERROR] /etc/enroot/hooks.d/98-nvidia.sh exited with return code 1
  slurmstepd: error: pyxis: couldn't start container
  slurmstepd: error: spank: required plugin spank_pyxis.so: task_init() failed with rc=-1
  slurmstepd: error: Failed to invoke spank plugin stack
  srun: error: 1x-instant-cluster-testing-1: task 0: Exited with exit code 1
  ```

  **Solution**: Downgrade `libnvidia-container-tools libnvidia-container1 nvidia-container-toolkit nvidia-container-toolkit-base` to `1.17.6-1`:

  ```bash
  sudo apt-get install -y \\
    libnvidia-container1=1.17.6-1 \\
    libnvidia-container-tools=1.17.6-1 \\
    nvidia-container-toolkit-base=1.17.6-1 \\
    nvidia-container-toolkit=1.17.6-1 \\
    --allow-downgrades --allow-change-held-packages
  ```

  https://github.com/NVIDIA/enroot/issues/232

## Enroot Health Check

[This script](https://github.com/datacrunch-research/cluster-job-orchestration/blob/main/enroot%2Bpyxis/enroot_check.sh) verifies that Enroot is correctly installed and operational on the host machine. It performs a systematic validation by checking Enroot's presence, confirming the version, importing a minimal Docker image (`ubuntu`), creating and launching a test container, and retrieving basic runtime information such as the container's PID and operating system details. It also ensures proper cleanup after execution, removing any residual test artifacts.

```bash
#!/bin/bash
set -e
echo "Checking Enroot installation..."
command -v enroot || { echo "Enroot not installed."; exit 1; }

echo "Running Enroot version check..."
enroot version || exit 1

TEST_IMG="ubuntu"
TEST_CONT="enroot_healthcheck"

[ -f ${TEST_IMG}.sqsh ] && rm -f ${TEST_IMG}.sqsh
enroot import docker://${TEST_IMG}
enroot create -n ${TEST_CONT} ${TEST_IMG}.sqsh
PID_OUTPUT=$(enroot start ${TEST_CONT} sh -c 'echo $$')
OS_OUTPUT=$(enroot start ${TEST_CONT} sh -c 'grep PRETTY /etc/os-release')

echo "PID: $PID_OUTPUT"
echo "OS: $OS_OUTPUT"

enroot remove -f ${TEST_CONT}
rm -f ${TEST_IMG}.sqsh
echo "Enroot health check PASSED."
```

## Pyxis Health Check

Here we have to alternatives:

1. `srun` command with pyxis flag:

```bash
srun --container-image=ubuntu grep PRETTY /etc/os-release
```

1. slurm script with SBATCH-extended flags:

```bash
#!/bin/bash
#SBATCH --job-name=pyxis_test
#SBATCH --output=pyxis_test.out
#SBATCH --container-name=pyxis_test
#SBATCH --container-image=docker://ubuntu

echo "Running inside container:"
grep PRETTY /etc/os-release
```

Now that we have tested and checked the incremental steps required, we will test an end-to-end training workload. First we will ensure the framework performs correctly in a single-node setup to move forward a multi-node one.

## Torchtitan single-node Testing

We perform a testing training of Llama 8B using the c4_test dataset from `tests/assets/c4_test` included in torchtitan repo. The container will perform a 10 steps training that will be reflected on the `/home/ubuntu/slurm_logging/headnode/%x_%j_headnode.err` file.

### Prerequisites: Torchtitan custom image

[torchtitan.dockerfile](https://github.com/datacrunch-research/cluster-job-orchestration/blob/main/enroot%2Bpyxis/torchtitan.dockerfile):

> Important: HF_TOKEN must be configured as environment variable

```bash
# Usinsg official Pytorch 2.7 + CUDA 12.8 base image
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime
#FROM ubuntu:22.04

ARG HF_TOKEN    
ENV HF_TOKEN=${HF_TOKEN}

# makes sure the shell used for subsequent RUN commands is exactly Bash, as located in /bin.
SHELL ["/bin/bash", "-c"]

# Install dependencies
# llamacpp gcc compilation tools
RUN apt-get update && apt-get install -y \\
    build-essential \\
    fzf \\
    ripgrep \\
    nvtop \\
    sudo \\
    kmod \\
    wget \\
    vim \\
    git \\
    curl \\
    bzip2 \\
    ca-certificates \\
    libglib2.0-0 \\
    libxext6 \\
    libsm6 \\
    libxrender1 \\
    libssl-dev \\
    libibverbs1 \\
    ibverbs-utils \\
    libmlx5-1 \\
    infiniband-diags
    # Cleanup command to remove the apt cache and reduce the image size: # IMPORTANT: Enforces using sudo apt update when entering the container
    #&& rm -rf /var/lib/apt/lists/*

# Cloning the repo
RUN git clone <https://github.com/pytorch/torchtitan>

# Change to the repo directory using WORKDIR
WORKDIR /workspace/torchtitan

RUN mkdir -p /root/.cache/huggingface

RUN pip install -r requirements.txt

# For CUDA 12.8 on worker nodes
RUN pip install torch torchvision torchaudio --index-url <https://download.pytorch.org/whl/cu128>.

# Download the tokenizer
RUN python3 scripts/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3.1-8B --tokenizer_path "original"      

# docker build -f torchtitan.dockerfile --build-arg HF_TOKEN="$HF_TOKEN" -t torchtitan_cuda128_torch27 .
# docker run --gpus all --shm-size 32g --network=host -v /home/ubuntu/.cache/huggingface:/root/.cache/huggingface --name torchtitan_workload -it --rm --ipc=host torchtitan_cuda128_torch27 bash -c 'CONFIG_FILE="./train_configs/llama3_8b.toml" ./run_llama_train.sh'
```

Build Docker image:

```
docker build -f torchtitan.dockerfile --build-arg HF_TOKEN="$HF_TOKEN" -t torchtitan_cuda128_torch27 .
```

Import image to Enroot:

With these schemes for the `enroot import` command, enroot can extract images from:

```bash
docker://[USER@][REGISTRY#]IMAGE[:TAG]  # Import a Docker image from a registry
dockerd://IMAGE[:TAG]                   # Import a Docker image from the Docker daemon
podman://IMAGE[:TAG]                    # Import a Docker image from a local podman repository
```

To import the previously created image to .sqsh we use:

```
enroot import -o /home/ubuntu/torchtitan_cuda128_torch27.sqsh dockerd://torchtitan_cuda128_torch27
```

[SLURM script](https://github.com/datacrunch-research/cluster-job-orchestration/blob/main/enroot%2Bpyxis/singlenode_torchtitan.sh):

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=50
#SBATCH --partition=gpus
#SBATCH --job-name=torchtitan_singlenode
#SBATCH -o /home/ubuntu/slurm_logging/headnode/%x_%j.out
#SBATCH -e /home/ubuntu/slurm_logging/headnode/%x_%j.err


CONFIG_FILE=${CONFIG_FILE:-"torchtitan/models/llama3/train_configs/debug_model.toml"}

srun --container-image=/home/ubuntu/torchtitan_cuda128_torch27.sqsh \
    --container-name=torchtitan_singlenode \
    --container-mounts=/home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
   --container-writable \
   --no-container-mount-home \
   bash run_train.sh --job.config_file ${CONFIG_FILE}
```

Pyxis flags included are:

- **`-container-image=/home/ubuntu/torchtitan_cuda128_torch27.sqsh`**

  Specifies the SquashFS file (or Enroot URI) to use as the container filesystem. In this case, we’re pointing at the `torchtitan_cuda128_torch27.sqsh` image previously created.

- **`-container-name=torchtitan-singlenode`**

  Name of the container. It will be cached in enroot data (in our config: `/mnt/local_disk/enroot/data/$UID`) as pyxis_torchtitan_singlenode.

- **`-container-mounts=/home/ubuntu/.cache/huggingface:/root/.cache/huggingface`**

  Bind-mounts our worker nodes Hugging Face cache directory into the container at `/root/.cache/huggingface`.

- **`-no-container-mount-home`**

  Prevents Pyxis from automatically bind-mounting our home directory into the container. Useful to isolate the container’s view of our home and avoid conflicts with NFS permissions.

- **`-container-writable`**

  Makes the container filesystem writable (by default SquashFS images are mounted read-only). Allows any in-container writes (e.g. installing packages, writing checkpoints) without additional mounts.

Resolve possible "Read-only file system" errors with:

```bash
#SBATCH --container-writable
```

For interactive use and debugging of the container created this command is proposed:

```bash
srun   \\
--partition=gpus   --gres=gpu:1   --ntasks=1   --cpus-per-task=4   \\
--container-image=/home/ubuntu/torchtitan_cuda128_torch27.sqsh   \\
--container-name=interactive_torchtitan   --pty bash
```

## Torchtitan Multi-node Testing

> The previously created `torchtitan_cuda128_torch27.sqsh` image is required

We now reproduce a fully production workload for a distributed training of the Llama 70B model with the c4 dataset.

[SLURM job configuration with Pyxis](https://github.com/datacrunch-research/cluster-job-orchestration/blob/main/enroot%2Bpyxis/multinode_torchtitan.sh):

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=50
#SBATCH --partition=gpus
#SBATCH --job-name=torchtitan_multinode
#SBATCH -o /home/ubuntu/slurm_logging/headnode/%x_%j.out
#SBATCH -e /home/ubuntu/slurm_logging/headnode/%x_%j.err

# === Compute these HOST-side ===
HEADNODE_HOST=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_ADDR=$(getent hosts "$HEADNODE_HOST" | grep -Eo '10\.[0-9]+\.[0-9]+\.[0-9]+' | head -n1)
MASTER_PORT=$((5000 + SLURM_JOB_ID % 10000))

CONFIG_FILE=${CONFIG_FILE:-"torchtitan/models/llama3/train_configs/llama3_70b.toml"}


echo "======== Distributed Config ========"
echo "HEADNODE_HOST: $HEADNODE_HOST"
echo "Resolved MASTER_ADDR: $MASTER_ADDR"
echo "Assigned MASTER_PORT: $MASTER_PORT"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "All Hosts:"
scontrol show hostnames "$SLURM_JOB_NODELIST"
echo "===================================="


# === Launch the container job ===
srun \
  --container-image=/home/ubuntu/torchtitan_cuda128_torch27.sqsh \
  --container-mounts=/home/ubuntu/.cache/huggingface:/root/.cache/huggingface \
  --container-writable \
  --export=ALL,HEADNODE_HOST=$HEADNODE_HOST,MASTER_ADDR=$MASTER_ADDR,MASTER_PORT=$MASTER_PORT,NCCL_DEBUG=INFO,NCCL_DEBUG_SUBSYS=ALL \
  torchrun --nnodes=2 --nproc_per_node=8 \
           --rdzv_backend=c10d \
           --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
           -m torchtitan.train --job.config_file ${CONFIG_FILE}
```

As some computation is required outside the container for resolving hostnames for the worker nodes and configuring the master address, we move the pyxis variables from the prelude to the `srun` command part.

For the SLURM script, two torchrun task are required in total: 1 task with 8 GPU per node. We assign output `-o` and error `-e` files for the head node. As the computation is performed inside the container (which doesn't address slurm) no log files for the worker nodes are provided (in the form of `srun --output=/home/ubuntu/slurm_logging/workernodes/multinode_torch_test_%j_node%N.out --error=/home/ubuntu/slurm_logging/workernodes/multinode_torch_test_%j_node%N.err`) Instead, the worker nodes logs will be printed out in these former files.

## References

[Cluster job orchestration datacrunch repo](https://github.com/datacrunch-research/cluster-job-orchestration)

**Slurm docs and scripts**

[SLURM Quick Start User Guide](https://slurm.schedmd.com/quickstart.html)

- [**SLURM simple guide**](https://github.com/datacrunch-research/microbenchmarking/blob/d28f998269e735a540797589e7916be59c478dec/slurm/multinode-slurm.md)
- https://github.com/pytorch/torchtitan/blob/main/multinode_trainer.slurm
- https://github.com/antferdom/simple-gpt/blob/master/gpt/run_multi_node.sh
- https://github.com/Yaofang-Liu/Mochi-Full-Finetuner/blob/main/src/genmo/mochi_preview/train_multi_nodes.sh
- https://docs.sglang.ai/references/multi_node.html
- [modal multinode training guide github](https://github.com/modal-labs/multinode-training-guide/tree/main)
- [Crusoe slurm cloud solution](https://crusoe.ai/blog/crusoe-cloud-slurm-solution/)

**Cluster health check**

- [cupy distributed test comms](https://github.com/cupy/cupy/blob/1794c835a086dda4cfb82d1152a495b61a82cbdc/tests/cupyx_tests/distributed_tests/test_comm.py#L20)
- [Imbue 70B infrastructure](https://imbue.com/research/70b-infrastructure/)
- [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering/tree/master)
- [Host-level health checks](https://github.com/imbue-ai/cluster-health/tree/master/health_checks)
- [Linux Performance Analysis in 60,000 Milliseconds, Netflix technical blogs](https://netflixtechblog.com/linux-performance-analysis-in-60-000-milliseconds-accc10403c55)
- [Pytorch Crusoe torchtitan](https://pytorch.org/blog/accelerating-training-float8-rowwise-crusoe/)

**Pyxis + Enroot**

- [enroot docs](https://github.com/NVIDIA/enroot/tree/master/doc)
- [Pyxis docs](https://github.com/NVIDIA/pyxis/wiki)
- [torchtitan example with pyxis](https://github.com/pytorch/torchtitan/issues/708)
- [scontrol within container Pyxis](https://github.com/NVIDIA/pyxis/issues/129)

