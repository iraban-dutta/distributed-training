import csv
import time
from pathlib import Path
import numpy as np
import torch


def sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_steps(trainer, warmup_steps: int, benchmark_steps: int) -> list[float]:
    """Return steady-state step times in seconds.

    Warmup runs are discarded. Each measured step is synchronized before
    the clock starts and after the step finishes, so the time includes
    the device work.
    """
    for _ in range(warmup_steps):
        trainer.train_step()
    sync_device(trainer.device)

    step_s = []
    for _ in range(benchmark_steps):
        sync_device(trainer.device)
        start = time.perf_counter()
        trainer.train_step()
        sync_device(trainer.device)
        step_s.append(time.perf_counter() - start)
    return step_s


def summarize(step_s: list[float], batch_size: int, seq_len: int) -> dict:
    arr = np.asarray(step_s, dtype=np.float64)
    median_s = float(np.median(arr))
    return {
        "steps": len(step_s),
        "median_step_ms": median_s * 1000,
        "mean_step_ms": float(arr.mean()) * 1000,
        "tokens_per_sec": (batch_size * seq_len) / median_s,
    }


def append_csv(path: str, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)