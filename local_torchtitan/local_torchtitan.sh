#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=50
#SBATCH --partition=gpus
#SBATCH --job-name=torchtitan_singlenode
#SBATCH -o /home/ubuntu/slurm_logging/headnode/%x_%j_headnode.out
#SBATCH -e /home/ubuntu/slurm_logging/headnode/%x_%j_headnode.err

CONFIG_FILE=${CONFIG_FILE:-"./torchtitan/models/llama3/train_configs/debug_model.toml"}
export CONFIG_FILE=${CONFIG_FILE}

cd /home/ubuntu/torchtitan

# adjust sbatch --ntasks and sbatch --nodes above and --nnodes below
# to your specific node count, and update target launch file.
srun --output=/home/ubuntu/slurm_logging/workernodes/%x_%j_node%N.out --error=/home/ubuntu/slurm_logging/workernodes/%x_%j_node%N.err \
 ./run_train.sh


# sbatch local_torchtitan.sh
