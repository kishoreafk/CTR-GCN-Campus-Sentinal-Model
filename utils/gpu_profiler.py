"""GPU profiling utilities for monitoring VRAM and compute utilization."""
import torch
import logging

log = logging.getLogger("gpu_profiler")


def log_gpu_stats(prefix: str = ""):
    """Log current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    max_allocated = torch.cuda.max_memory_allocated() / 1e9
    log.info(f"{prefix}GPU Memory: allocated={allocated:.2f}GB, "
             f"reserved={reserved:.2f}GB, max_allocated={max_allocated:.2f}GB")


def get_gpu_info() -> dict:
    """Get GPU device information."""
    if not torch.cuda.is_available():
        return {"available": False}

    p = torch.cuda.get_device_properties(0)
    return {
        "available": True,
        "name": p.name,
        "vram_gb": p.total_memory / 1e9,
        "sm": f"{p.major}.{p.minor}",
        "bf16": torch.cuda.is_bf16_supported(),
        "allocated_gb": torch.cuda.memory_allocated() / 1e9,
        "reserved_gb": torch.cuda.memory_reserved() / 1e9,
    }


class GPUProfiler:
    """Context manager for profiling GPU memory usage of a code block."""

    def __init__(self, name: str = "block"):
        self.name = name
        self.start_mem = 0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.start_mem = torch.cuda.memory_allocated()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            end_mem = torch.cuda.memory_allocated()
            delta = (end_mem - self.start_mem) / 1e6
            log.info(f"[{self.name}] GPU memory delta: {delta:+.1f} MB")
