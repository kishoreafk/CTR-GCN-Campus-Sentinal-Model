"""
Pre-flight checks run before any expensive operation.
Raises RuntimeError on critical failures.
"""
import shutil, logging
import torch
log = logging.getLogger("system_checks")

def check_disk_space(required_gb: float, path: str = ".") -> None:
    free = shutil.disk_usage(path).free / 1e9
    if free < required_gb:
        raise RuntimeError(
            f"Need {required_gb:.0f} GB free at '{path}', have {free:.1f} GB")
    log.info(f"Disk OK: {free:.1f} GB free (need {required_gb:.0f} GB)")

def check_gpu() -> dict:
    assert torch.cuda.is_available(), "CUDA not available"
    p = torch.cuda.get_device_properties(0)
    info = {
        "name": p.name,
        "vram_gb": p.total_memory / 1e9,
        "sm": f"{p.major}.{p.minor}",
        "bf16": torch.cuda.is_bf16_supported(),
    }
    assert p.major >= 8, f"Need SM 8.x+, got {info['sm']}"
    log.info(f"GPU: {info['name']}  VRAM: {info['vram_gb']:.1f} GB  "
             f"SM: {info['sm']}  BF16: {info['bf16']}")
    return info

def preflight(required_disk_gb: float = 50.0) -> None:
    check_gpu()
    check_disk_space(required_disk_gb)
