# torchrun --standalone --nproc_per_node=8 nccl_torch.py --min-size 1MB --max-size 128MB --num-iters 10 --profile --output "results10_128"

# TORCH_DISTRIBUTED_DEBUG=DETAIL
# NCCL_DEBUG=INFO
# NCCL_BLOCKING_WAIT=1 For silent hangs


import torch
import torch.distributed as dist
import typer
import os, time
from typing import List, Optional

def main(
    backend: str = typer.Option("nccl", help="Distributed backend (nccl/gloo). Use NCCL for GPUs."),
    num_iters: int = typer.Option(20, help="Iterations per message size to measure."),
    min_size: str = typer.Option("1KB", help="Minimum message size (e.g., 1KB, 1MB)."),
    max_size: str = typer.Option("1GB", help="Maximum message size to test."),
    sizes: Optional[List[str]] = typer.Option(None, help="Explicit list of message sizes to test (overrides min/max)."),
    profile: bool = typer.Option(False, help="Enable profiling of the all-reduce operations."),
    output: str = typer.Option("", help="File to write results (optional).")
):
    """NCCL All-Reduce Benchmark"""
    
    # Initialize distributed process group
    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Set current CUDA device based on local rank (if using NCCL backend on GPUs)
    # If LOCAL_RANK is provided (torchrun sets this), use it, otherwise use rank % #GPUs as fallback.
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

    # Utility to parse size strings integer bytes
    def parse_size(size_str):
        size_str = size_str.strip().upper()
        if size_str.endswith("KB"):
            return int(size_str[:-2]) * 1024
        if size_str.endswith("MB"):
            return int(size_str[:-2]) * 1024**2
        if size_str.endswith("GB"):
            return int(size_str[:-2]) * 1024**3
        return int(size_str)  # if just a number of bytes

    if sizes:
        # Use explicit sizes provided
        sizes_bytes = [parse_size(s) for s in sizes]
    else:
        min_bytes = parse_size(min_size)
        max_bytes = parse_size(max_size)
        if min_bytes < 1: 
            min_bytes = 1
        # Generate sizes in a geometric progression from min to max (x2)
        sizes_bytes = []
        size = min_bytes
        while size <= max_bytes:
            sizes_bytes.append(size)
            size *= 2  # gemoetric progression, could be impl as log-scale steps
        if sizes_bytes[-1] != max_bytes:
            sizes_bytes.append(max_bytes)

    import numpy as np

    # Arrays to accumulate results (for rank 0)
    latencies_avg, latencies_min, latencies_max = [], [], [] # latency (s) per size
    bandwidths_avg, bandwidths_min, bandwidths_max = [], [], [] # bandwidth (GB/s) per size

    for size in sizes_bytes:
        # Create tensor of given size
        num_elems = size // 2  # bf16, 4 if fp32
        tensor = torch.ones(num_elems, dtype=torch.bfloat16).cuda()

        # Warm-up
        for i in range(2):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        if profile and rank == 0:
            # Only rank 0 profiles to reduce overhead
            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            profiler = torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=False)
            profiler.__enter__()  # start profiling
            prof_enabled = True
        else:
            prof_enabled = False

        # Time multiple all-reduce iterations
        start = time.time()
        for i in range(num_iters):
            dist.all_reduce(tensor)  # sum reduction by default
            # Synchronize to wait for each allreduce to complete before the next iteration
            torch.cuda.synchronize()
        elapsed = time.time() - start
        if prof_enabled:
            profiler.step()  # record until here

        avg_time = elapsed / num_iters
        # Gather timings from all ranks to rank 0 with dist.gather
        t_tensor = torch.tensor(avg_time, dtype=torch.float64).cuda().squeeze() # dist.gather expect a scalar tensor
        if rank == 0:
            all_times = [torch.empty_like(t_tensor) for _ in range(world_size)]
            dist.gather(t_tensor, gather_list=all_times, dst=0)
            times = [t.item() for t in all_times]
            t_avg = sum(times) / len(times)
            t_min = min(times)
            t_max = max(times)
            # Compute bandwidth per rank (GB/s per GPU) for stats
            # Bandwidth = data_size_bytes / time_per_gpu. Using per-GPU data_size.
            gb = size / 1e9  # bytes to GB
            bw_avg = gb / t_avg
            bw_min = gb / t_max  # t_max gives lowest bandwidth
            bw_max = gb / t_min  # t_min gives highest bandwidth

            latencies_avg.append(t_avg)
            latencies_min.append(t_min)
            latencies_max.append(t_max)

            bandwidths_avg.append(bw_avg)
            bandwidths_min.append(bw_min)
            bandwidths_max.append(bw_max)
            print(f"[Size {size} bytes] All-reduce latency: {t_avg*1e3:.3f} ms (min {t_min*1e3:.3f}, max {t_max*1e3:.3f}); "
                  f"Bandwidth: {bw_avg:.3f} GB/s (min {bw_min:.3f}, max {bw_max:.3f})")
        else:
            # Non-zero ranks just send their data
            dist.gather(t_tensor, dst=0)

    home_dir = os.environ.get("HOME")

    # Save profiler results
    if prof_enabled:
        profiler.__exit__(None, None, None)
        # Save profiler results to a trace file
        profiler_path = f"{home_dir}/profile_rank{rank}.json"
        profiler.export_chrome_trace(f"{profiler_path}")
        print(f"Profiler trace saved for rank {rank} in {profiler_path}")

    # Save results 
    if output and rank == 0:
        import csv
        output_path = f"{home_dir}/{output}"
        with open(f"{output_path}", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Size(bytes)", 
                             "Latency_avg(s)", "Latency_min(s)", "Latency_max(s)", 
                             "Bandwidth_avg(GB/s)", "Bandwidth_min(GB/s)", "Bandwidth_max(GB/s)"])
            for i, size in enumerate(sizes_bytes):
                writer.writerow([
                    size, 
                    latencies_avg[i], latencies_min[i], latencies_max[i], 
                    bandwidths_avg[i], bandwidths_min[i], bandwidths_max[i]
                ])
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)

