import os
import sys
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def all_reduce_latency(nbytes, rank):
  buf = torch.randn(nbytes // 4).cuda(rank)

  torch.cuda.synchronize(rank)
  for _ in range(5): dist.all_reduce(buf)
  torch.cuda.synchronize(rank)

  # run 25 times
  torch.cuda.synchronize(rank)
  begin = time.perf_counter()
  for _ in range(25): dist.all_reduce(buf)
  torch.cuda.synchronize(rank)

  return (time.perf_counter() - begin) / 25

def run(rank, size):
  print(f"hello from {rank}")
  sz = 256_000_000 # 256 MB
  for i in range(10):
    speed = all_reduce_latency(sz, rank)
    print(f"{rank}: {speed*1e3:.2f} ms to copy {sz*1e-6:.2f} MB, {sz*1e-9/speed:.2f} GB/s")

def init_process(rank, size, fn, backend='nccl'):
  os.environ['MASTER_ADDR'] = '127.0.0.1'
  os.environ['MASTER_PORT'] = '29500'
  dist.init_process_group(backend, rank=rank, world_size=size)
  fn(rank, size)

if __name__ == "__main__":
  size = int(sys.argv[1])
  processes = []
  mp.set_start_method("spawn")

  for rank in range(size):
    p = mp.Process(target=init_process, args=(rank, size, run))
    p.start()
    processes.append(p)

  for p in processes:
    p.join()