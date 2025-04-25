For manual torchrun testing just:
1. run workernode1.sh on node1
2. run workernode2.sh on node2

For SLURM just:
```bash 
sbatch slurm_multinode_torch_distributed.sh
```
Ensure logging paths are created or modified correctly.
Ensure configuration of torchrun and SLURM is properly tuned (Current: 2 nodes with 8 gpus)
