
# NCCL All‑Reduce benchmark with optional tensor pre‑allocation
 #------------------------------------------------------------
 # Usage example (8 GPUs):
# torchrun --standalone --nproc_per_node=8 nccl_torch.py --min-size 512MB --max-size 8GB --num-iters 5 --pin-memory --preallocate
 
 # Set --preallocate true (default) to allocate each tensor **once** and reuse it
 # across warm‑up and timed iterations, ensuring that no allocation or host→device
 # copy contaminates the timing window.

import os, time, json
from pathlib import Path
from typing import List, Optional

import torch
import torch.distributed as dist
import typer

###############################################################################
# Utility helpers                                                              
###############################################################################

def load_system_config():
    """Load the system configuration with theoretical values (if present)."""
    config_paths = [
        "generated_config.json",
        "../health_check/generated_config.json",
        "../../health_check/generated_config.json",
        os.path.expanduser("~/generated_config.json"),
    ]
    for path in config_paths:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    print("System config not found. Run health_check/generate_config_from_system.py first for theoretical comparisons.")
    return None


def calculate_bandwidths(size_bytes: int, time_seconds: float, world_size: int):
    """Return (algorithm BW, bus BW) in GB/s."""
    alg_bw = size_bytes / 1e9 / time_seconds
    factor = (2 * (world_size - 1)) / world_size  # ring all‑reduce traffic factor
    bus_bw = alg_bw * factor
    return alg_bw, bus_bw


def print_performance_analysis(alg_bw: float, bus_bw: float, config: dict, world_size: int):
    if not config or "nvlink" not in config:
        print("  No theoretical values available for comparison")
        return
    nvlink_info = config["nvlink"]
    total_nvlink_bw = nvlink_info.get("total_nvlink_bandwidth_per_gpu", 0)
    if total_nvlink_bw <= 0:
        return
    bus_factor = (2 * (world_size - 1)) / world_size
    theoretical_bus_bw = total_nvlink_bw * bus_factor
    bus_eff = bus_bw / theoretical_bus_bw * 100
    nv_util = alg_bw / total_nvlink_bw * 100
    print(f"  Bus Efficiency: {bus_eff:.1f}% of theoretical max bus bandwidth: {theoretical_bus_bw:.1f} GB/s")
    print(f"  NVLink Utilization: {nv_util:.1f}% of theoretical max NVLink bandwidth: {total_nvlink_bw:.1f} GB/s")
    print(f"  Theoretical bus bandwidth calculation: {total_nvlink_bw:.1f} GB/s × {bus_factor:.3f} (factor for {world_size} GPUs) = {theoretical_bus_bw:.1f} GB/s")

###############################################################################
# Main benchmark                                                               
###############################################################################

def main(
    backend: str = typer.Option("nccl", help="Distributed backend (nccl/gloo)."),
    num_iters: int = typer.Option(20, help="Iterations per message size."),
    min_size: str = typer.Option("1KB"),
    max_size: str = typer.Option("1GB"),
    sizes: Optional[List[str]] = typer.Option(None, help="Explicit list of message sizes (overrides min/max)."),
    profile: bool = typer.Option(False, help="Enable torch.profiler."),
    output: str = typer.Option("", help="CSV file name for results."),
    config_path: str = typer.Option("", help="Path to system‑config JSON."),
    dtype: str = typer.Option("bfloat16", help="float32 | float16 | bfloat16"),
    warmup_iters: int = typer.Option(5),
    pin_memory: bool = typer.Option(False),
    preallocate: bool = typer.Option(True, help="Pre‑allocate tensors outside the timed region"),
):
    """NCCL all‑reduce micro‑benchmark with optional tensor pre‑allocation."""

    # ---------------------------------------------------------------------
    # Config / distributed init
    # ---------------------------------------------------------------------
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            system_config = json.load(f)
    else:
        system_config = load_system_config()

    dist.init_process_group(backend=backend)
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    torch.cuda.set_device(local_rank)

    # ---------------------------------------------------------------------
    # Print once per job
    # ---------------------------------------------------------------------
    if rank == 0 and system_config:
        nv = system_config.get("nvlink", {})
        print("System Information:")
        print(f"  GPU Model: {nv.get('gpu_model', 'Unknown')}")
        print(f"  GPU Count: {nv.get('gpu_count', 'Unknown')}")
        print(f"  NVLink per GPU: {nv.get('links_per_gpu', 'Unknown')} links at {nv.get('link_speed_gb_s', 'Unknown')} GB/s")
        print(f"  Total NVLink BW: {nv.get('total_nvlink_bandwidth_per_gpu', 'Unknown')} GB/s per GPU")
        print(f"  World size: {world_size} GPUs")
        print("Test Configuration:")
        print(f"  Warmup iterations: {warmup_iters}")
        print(f"  Pinned memory: {pin_memory}")
        print(f"  Pre‑allocate tensors: {preallocate}")
        print()

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------
    def parse_size(sz: str):
        s = sz.strip().upper()
        if s.endswith("KB"):
            return int(s[:-2]) * 1024
        if s.endswith("MB"):
            return int(s[:-2]) * 1024**2
        if s.endswith("GB"):
            return int(s[:-2]) * 1024**3
        return int(s)

    if sizes:
        sizes_bytes = [parse_size(s) for s in sizes]
    else:
        min_b, max_b = parse_size(min_size), parse_size(max_size)
        min_b = max(min_b, 1)
        sizes_bytes = []
        s = min_b
        while s <= max_b:
            sizes_bytes.append(s)
            s *= 2
        if sizes_bytes[-1] != max_b:
            sizes_bytes.append(max_b)

    # ---------------------------------------------------------------------
    # DType setup
    # ---------------------------------------------------------------------
    dtype_map = {
        "float32": (torch.float32, 4),
        "float16": (torch.float16, 2),
        "bfloat16": (torch.bfloat16, 2),
    }
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}")
    torch_dtype, bytes_per_elem = dtype_map[dtype]

    # ---------------------------------------------------------------------
    # **Pre‑allocation bucket**
    # ---------------------------------------------------------------------
    preallocated_tensors = {} if preallocate else None  # <‑‑ stores tensor per size

    # Accumulators (rank‑0 only)
    lat_avg = []
    alg_bw_avg = []
    bus_bw_avg = []

    for size in sizes_bytes:
        num_elems = size // bytes_per_elem

        # --------------------------------------------------------------
        # Retrieve or create tensor
        # --------------------------------------------------------------
        if preallocate and size in preallocated_tensors:
            tensor = preallocated_tensors[size]  # <‑‑ pre‑allocated tensor reused
        else:
            init_fn = torch.zeros if dtype == "float32" else torch.ones  # arbitrary data
            if pin_memory:
                cpu_buf = init_fn(num_elems, dtype=torch_dtype, pin_memory=True)
                tensor = cpu_buf.cuda()
            else:
                tensor = init_fn(num_elems, dtype=torch_dtype, device="cuda")
            if preallocate:
                preallocated_tensors[size] = tensor  # store for future reuse

        # --------------------------------------------------------------
        # Warm‑up
        # --------------------------------------------------------------
        for _ in range(warmup_iters):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()

        # --------------------------------------------------------------
        # (Optional) profiler
        # --------------------------------------------------------------
        prof_ctx = (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            )
            if profile and rank == 0
            else None
        )
        if prof_ctx:
            prof_ctx.__enter__()

        # --------------------------------------------------------------
        # Timed section
        # --------------------------------------------------------------
        start = time.time()
        for _ in range(num_iters):
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
        elapsed = time.time() - start

        if prof_ctx:
            prof_ctx.step()

        # --------------------------------------------------------------
        # Gather times (rank‑0)
        # --------------------------------------------------------------
        t_tensor = torch.tensor(elapsed / num_iters, dtype=torch.float64, device="cuda")
        if rank == 0:
            recv = [torch.empty_like(t_tensor) for _ in range(world_size)]
            dist.gather(t_tensor, gather_list=recv, dst=0)
            times = [t.item() for t in recv]
            t_avg, t_min, t_max = sum(times) / len(times), min(times), max(times)
            a_bw_avg, b_bw_avg = calculate_bandwidths(size, t_avg, world_size)
            lat_avg.append(t_avg)
            alg_bw_avg.append(a_bw_avg)
            bus_bw_avg.append(b_bw_avg)

            print(f"[Size {size} bytes]")
            print(f"  Latency: {t_avg*1e3:.3f} ms (min {t_min*1e3:.3f}, max {t_max*1e3:.3f})")
            print(f"  Alg BW : {a_bw_avg:.3f} GB/s")
            print(f"  Bus BW : {b_bw_avg:.3f} GB/s")
            print_performance_analysis(a_bw_avg, b_bw_avg, system_config, world_size)
            print()
        else:
            dist.gather(t_tensor, dst=0)

        if prof_ctx:
            prof_ctx.__exit__(None, None, None)
            path = Path(os.environ.get("HOME", ".")) / f"profile_rank{rank}_size{size}.json"
            prof_ctx.export_chrome_trace(str(path))

    # ---------------------------------------------------------------------
    # CSV output (rank‑0)
    # ---------------------------------------------------------------------
    if output and rank == 0:
        out_path = Path(os.environ.get("HOME", ".")) / output
        import csv
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["size(B)", "latency_avg(s)", "alg_bw_avg(GB/s)", "bus_bw_avg(GB/s)"])
            for s, l, a, b in zip(sizes_bytes, lat_avg, alg_bw_avg, bus_bw_avg):
                w.writerow([s, l, a, b])
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    typer.run(main)
