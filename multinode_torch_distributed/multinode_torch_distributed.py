# torchrun_benchmark.py
import os
import time
import torch
import torch.distributed as dist

def all_reduce_latency(nbytes):
    # Each process has already selected the correct GPU via `cuda.set_device`
    buf = torch.randn(nbytes // 4, device='cuda')

    torch.cuda.synchronize()
    for _ in range(5):
        dist.all_reduce(buf)
    torch.cuda.synchronize()

    begin = time.perf_counter()
    for _ in range(25):
        dist.all_reduce(buf)
    torch.cuda.synchronize()

    return (time.perf_counter() - begin) / 25

def main():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    print(f"[Rank {rank}/{world_size}] running on local GPU {local_rank} on host {os.uname()[1]}")
    sz = 256_000_000  # 256 MB

    for i in range(10):
        speed = all_reduce_latency(sz)
        print(f"[Rank {rank}] {speed*1e3:.2f} ms for {sz*1e-6:.2f} MB → {sz*1e-9/speed:.2f} GB/s")

    dist.destroy_process_group()

if __name__ == "__main__":
    dist.init_process_group(backend="nccl", init_method="env://")
    main()
