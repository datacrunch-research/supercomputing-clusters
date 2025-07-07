# torchrun --standalone --nproc_per_node=8 nccl_torch.py --min-size 1MB --max-size 128MB --num-iters 10 --profile --output "results10_128"

# TORCH_DISTRIBUTED_DEBUG=DETAIL
# NCCL_DEBUG=INFO
# NCCL_BLOCKING_WAIT=1 For silent hangs


import torch
import torch.distributed as dist
import typer
import os, time, json
from typing import List, Optional
from pathlib import Path

def load_system_config():
    """Load the system configuration with theoretical values"""
    # Look for config in several locations
    config_paths = [
        "generated_config.json",
        "../health_check/generated_config.json", 
        "../../health_check/generated_config.json",
        os.path.expanduser("~/generated_config.json")
    ]
    
    for path in config_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    
    print( "System config not found. Run generate_config_from_system.py first for theoretical comparisons.")
    return None

def calculate_bandwidths(size_bytes: int, time_seconds: float, world_size: int):
    """
    Calculate both algorithm and bus bandwidth
    
    Algorithm BW: Effective bandwidth from application perspective
    Bus BW: Actual interconnect traffic (includes AllReduce overhead factor)
    """
    # Algorithm bandwidth
    alg_bw = size_bytes / 1e9 / time_seconds
    
    # Bus bandwidth 
    # Factor = 2*(N-1)/N for ring AllReduce algorithm
    factor = (2 * (world_size - 1)) / world_size
    bus_bw = alg_bw * factor
    
    return alg_bw, bus_bw

def print_performance_analysis(alg_bw: float, bus_bw: float, config: dict, world_size: int):
    """Print performance analysis compared to theoretical values"""
    if not config or 'nvlink' not in config:
        print(f"  No theoretical values available for comparison")
        return
    
    nvlink_info = config['nvlink']
    
    # Get raw hardware values
    total_nvlink_bw = nvlink_info.get('total_nvlink_bandwidth_per_gpu', 0)
    
    # Calculate theoretical bus bandwidth based on actual world_size
    if total_nvlink_bw > 0:
        bus_factor = (2 * (world_size - 1)) / world_size  # Ring AllReduce bus traffic factor
        theoretical_bus_bw = total_nvlink_bw * bus_factor
        
        bus_efficiency = (bus_bw / theoretical_bus_bw) * 100
        print(f"  Bus Efficiency: {bus_efficiency:.1f}% of theoretical max bus bandwidth: {theoretical_bus_bw:.1f} GB/s")
        
        nvlink_utilization = (alg_bw / total_nvlink_bw) * 100
        print(f"  NVLink Utilization: {nvlink_utilization:.1f}% of theoretical max NVLink bandwidth: {total_nvlink_bw:.1f} GB/s")
        print(f"  Theoretical bus bandwidth calculation: {total_nvlink_bw:.1f} GB/s × {bus_factor:.3f} (factor for {world_size} GPUs) = {theoretical_bus_bw:.1f} GB/s")

def main(
    backend: str = typer.Option("nccl", help="Distributed backend (nccl/gloo). Use NCCL for GPUs."),
    num_iters: int = typer.Option(20, help="Iterations per message size to measure."),
    min_size: str = typer.Option("1KB", help="Minimum message size (e.g., 1KB, 1MB)."),
    max_size: str = typer.Option("1GB", help="Maximum message size to test."),
    sizes: Optional[List[str]] = typer.Option(None, help="Explicit list of message sizes to test (overrides min/max)."),
    profile: bool = typer.Option(False, help="Enable profiling of the all-reduce operations."),
    output: str = typer.Option("", help="File to write results (optional)."),
    config_path: str = typer.Option("", help="Path to system config file (auto-detected if not specified).")
):
    """NCCL All-Reduce Benchmark with Theoretical Performance Comparison"""
    
    # Load system configuration for theoretical comparisons
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            system_config = json.load(f)
    else:
        system_config = load_system_config()
    
    # Initialize distributed process group
    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # Set current CUDA device based on local rank (if using NCCL backend on GPUs)
    # If LOCAL_RANK is provided (torchrun sets this), use it, otherwise use rank % #GPUs as fallback.
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

    # Print system info on rank 0
    if rank == 0 and system_config:
        nvlink_info = system_config.get('nvlink', {})
        print(f"System Information:")
        print(f"  GPU Model: {nvlink_info.get('gpu_model', 'Unknown')}")
        print(f"  GPU Count: {nvlink_info.get('gpu_count', 'Unknown')}")
        print(f"  NVLink per GPU: {nvlink_info.get('links_per_gpu', 'Unknown')} links at {nvlink_info.get('link_speed_gb_s', 'Unknown')} GB/s")
        print(f"  Total NVLink BW: {nvlink_info.get('total_nvlink_bandwidth_per_gpu', 'Unknown')} GB/s per GPU")
        print(f"  World size: {world_size} GPUs")
        print()

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
    alg_bandwidths_avg, alg_bandwidths_min, alg_bandwidths_max = [], [], [] # algorithm bandwidth (GB/s) per size
    bus_bandwidths_avg, bus_bandwidths_min, bus_bandwidths_max = [], [], [] # bus bandwidth (GB/s) per size

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
            
            # Compute both algorithm and bus bandwidth
            alg_bw_avg, bus_bw_avg = calculate_bandwidths(size, t_avg, world_size)
            alg_bw_min, bus_bw_min = calculate_bandwidths(size, t_max, world_size)  # t_max gives lowest bandwidth
            alg_bw_max, bus_bw_max = calculate_bandwidths(size, t_min, world_size)  # t_min gives highest bandwidth

            latencies_avg.append(t_avg)
            latencies_min.append(t_min)
            latencies_max.append(t_max)

            alg_bandwidths_avg.append(alg_bw_avg)
            alg_bandwidths_min.append(alg_bw_min)
            alg_bandwidths_max.append(alg_bw_max)
            
            bus_bandwidths_avg.append(bus_bw_avg)
            bus_bandwidths_min.append(bus_bw_min)
            bus_bandwidths_max.append(bus_bw_max)
            
            print(f"[Size {size} bytes] \n  All-reduce latency: {t_avg*1e3:.3f} ms (min {t_min*1e3:.3f}, max {t_max*1e3:.3f})")
            print(f"  Algorithm BW: {alg_bw_avg:.3f} GB/s (min {alg_bw_min:.3f}, max {alg_bw_max:.3f})")
            print(f"  Bus BW: {bus_bw_avg:.3f} GB/s (min {bus_bw_min:.3f}, max {bus_bw_max:.3f})")
            
            # Print performance analysis
            print_performance_analysis(alg_bw_avg, bus_bw_avg, system_config, world_size)
            print() # empty line
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
                             "Algorithm_BW_avg(GB/s)", "Algorithm_BW_min(GB/s)", "Algorithm_BW_max(GB/s)",
                             "Bus_BW_avg(GB/s)", "Bus_BW_min(GB/s)", "Bus_BW_max(GB/s)"])
            for i, size in enumerate(sizes_bytes):
                writer.writerow([
                    size, 
                    latencies_avg[i], latencies_min[i], latencies_max[i], 
                    alg_bandwidths_avg[i], alg_bandwidths_min[i], alg_bandwidths_max[i],
                    bus_bandwidths_avg[i], bus_bandwidths_min[i], bus_bandwidths_max[i]
                ])
        print(f"Results saved to {output_path}")


if __name__ == "__main__":
    typer.run(main)

