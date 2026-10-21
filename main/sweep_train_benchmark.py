import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import importlib
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def resolve_device(device:str) -> tuple[int, int]:
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda":
        WARMUP, BENCHMARK = 50, 50
    else:
        WARMUP, BENCHMARK = 5, 50

    return WARMUP, BENCHMARK



# ========================================================
# Global settings
# ========================================================

# --------------------------------
# Part1: Config Settings
# --------------------------------
ROOT = Path(__file__).resolve().parents[1]
# CONFIG = "config.local"
CONFIG = "config.dense_default"
_cfg = importlib.import_module(CONFIG)

# --------------------------------
# Part2: Benchmark Settings
# --------------------------------
TO_BENCHMARK = False
WARMUP, BENCHMARK = resolve_device(_cfg.DEVICE)
B_SWEEP = [_cfg.trainer_config.batch_size * m for m in (1, 2, 4)]
T_SWEEP = [_cfg.llm_config.ctx_len * m for m in (1, 2, 4)]
GRAD_ACCUM = 4

# --------------------------------
# Part3: Profiling Settings
# --------------------------------
TO_PROFILE = True
PROFILE_TRACES = False # True: trace and operators.txt. False: memory.html.
if PROFILE_TRACES:
    PROFILE_WAIT, PROFILE_WARMUP, PROFILE_ACTIVE = 25, 25, 25
else:
    PROFILE_WAIT, PROFILE_WARMUP, PROFILE_ACTIVE = 0, 0, 4


# # ========================================================
# # Settings: CPU Baseline
# # ========================================================
# CPU = {
#     "ctx_len": _cfg.llm_config.ctx_len,
#     "use_flash": True,
#     "use_checkpoint": False,
#     "batch_size": _cfg.trainer_config.batch_size,
#     "grad_accum": 1,
#     "precision": "fp32",
#     "compile_model": False,
#     "optimizer": "adamw",
#     "fused_optimizer": False,
# }

# # ========================================================
# # RUNS - CPU
# # ========================================================
# RUNS = [
#     {"name": "baseline", **CPU},
#     {"name": "checkpoint", **CPU, "use_checkpoint": True},
#     {"name": "accum", **CPU, "batch_size": (_cfg.trainer_config.batch_size//GRAD_ACCUM), "grad_accum": 4},
#     {"name": "compile", **CPU, "compile_model": True},
#     {"name": "manual_attn", **CPU, "use_flash": False},
#     {"name": "compile+checkpoint", **CPU, "compile_model": True, "use_checkpoint": True},
#     {"name": "checkpoint+manual_attn", **CPU, "use_checkpoint": True, "use_flash": False},

#     # To sweep B or T, add a sweep key as shown below:
#     {"name": "baseline_B", **CPU, "sweep": "B"},
#     {"name": "baseline_T", **CPU, "sweep": "T"},
# ]


# # ========================================================
# # CUDA memory plots. 
# #   Leave empty on CPU.
# # ========================================================
# PROFILE = []


# ========================================================
# Settings: GPU Baseline
# ========================================================
_cfg = importlib.import_module(CONFIG)
GPU = {
    "ctx_len": _cfg.llm_config.ctx_len,
    "use_flash": False,
    "use_checkpoint": False,
    "batch_size": _cfg.trainer_config.batch_size,
    "grad_accum": 1,
    "precision": "fp32",
    "compile_model": False,
    "optimizer": "adamw",
    "fused_optimizer": False,
}

# ========================================================
# RUNS - GPU
#   1. Run1 is the baseline. 
#   2. Run2–Run7 each change one field.
#   3. Run8–Run10 stack levers.
# ========================================================

RUNS = [
    {"name": "baseline", **GPU},
    {"name": "flash", **GPU, "use_flash": True},
    {"name": "checkpoint", **GPU, "use_checkpoint": True},
    {"name": "grad_accum", **GPU, "batch_size": (_cfg.trainer_config.batch_size//GRAD_ACCUM), "grad_accum": GRAD_ACCUM},
    {"name": "bf16", **GPU, "precision": "bf16"},
    {"name": "compile", **GPU, "compile_model": True},
    {"name": "fused", **GPU, "fused_optimizer": True},
    {
        "name": "flash+compile+fused",
        **GPU,
        "use_flash": True,
        "compile_model": True,
        "fused_optimizer": True,
    },
    {
        "name": "flash+compile+fused+bf16",
        **GPU,
        "use_flash": True,
        "compile_model": True,
        "fused_optimizer": True,
        "precision": "bf16",
    },
    {
        "name": "all",
        **GPU,
        "use_flash": True,
        "compile_model": True,
        "fused_optimizer": True,
        "precision": "bf16",
        "use_checkpoint": True,
        "batch_size": (_cfg.trainer_config.batch_size//GRAD_ACCUM),
        "grad_accum": GRAD_ACCUM,
    },

    # Sweeps: Batch Size & Context length
    {"name": "baseline_B", **GPU, "sweep": "B"},
    {"name": "baseline_T", **GPU, "sweep": "T"},
    {
        "name": "flash+compile+fused+bf16_B",
        **GPU,
        "use_flash": True,
        "compile_model": True,
        "fused_optimizer": True,
        "precision": "bf16",
        "sweep": "B",
    },
    {
        "name": "flash+compile+fused+bf16_T",
        **GPU,
        "use_flash": True,
        "compile_model": True,
        "fused_optimizer": True,
        "precision": "bf16",
        "sweep": "T",
    },

]

# ========================================================
# CUDA memory plots. 
#   Memory timelines for checkpointing, accumulation, and precision.
# ========================================================
PROFILE = [
    "baseline",
    "checkpoint",
    "grad_accum",
    "bf16",
]



def points(run):
    sweep = run.get("sweep")
    if sweep == "B":
        return [(f"{run['name']} B={b}", b, run["ctx_len"]) for b in B_SWEEP]
    if sweep == "T":
        return [(f"{run['name']} T={t}", run["batch_size"], t) for t in T_SWEEP]
    return [(run["name"], run["batch_size"], run["ctx_len"])]


def command(run, b, t, csv_path):
    return [
        sys.executable, "-m", "main.train_benchmark",
        "--config", CONFIG,
        "--benchmark_csv", str(csv_path),
        "--warmup", str(WARMUP),
        "--benchmark", str(BENCHMARK),
        "--batch_size", str(b),
        "--ctx_len", str(t),
        "--grad_accum", str(run["grad_accum"]),
        "--precision", run["precision"],
        "--compile_model", str(run["compile_model"]).lower(),
        "--use_activation_checkpoint", str(run["use_checkpoint"]).lower(),
        "--use_flash", str(run["use_flash"]).lower(),
        "--optimizer", run["optimizer"],
        "--fused_optimizer", str(run["fused_optimizer"]).lower(),
    ]

def main():
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    csv_path = ROOT / "runs" / "sweeps" / stamp / "results.csv"
    csv_path.parent.mkdir(parents=True)
    labels = []
    rates = []

    # Benchmark Runs
    if TO_BENCHMARK:
        rows = []
        for run in RUNS:
            for label, b, t in points(run):
                print(label)
                subprocess.run(command(run, b, t, csv_path), check=True, cwd=ROOT)
                with csv_path.open(newline="") as f:
                    rate = float(list(csv.DictReader(f))[-1]["tokens_per_sec"])
                rows.append({
                    "name": run["name"],
                    "sweep": run.get("sweep"),
                    "b": b,
                    "t": t,
                    "label": label,
                    "rate": rate,
                })

        # Plot - Point Runs
        bars = [r for r in rows if r["sweep"] is None]
        if bars:
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.bar([r["label"] for r in bars], [r["rate"] for r in bars])
            ax.set_ylabel("tokens/sec")
            ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            plot_path = csv_path.parent / "tokens_per_sec.png"
            fig.savefig(plot_path)
            print(plot_path)


        def save_lines(sweep, xkey, xlabel, filename):
            chosen = [r for r in rows if r["sweep"] == sweep]
            if not chosen:
                return
            names = []
            for r in chosen:
                if r["name"] not in names:
                    names.append(r["name"])
            fig, ax = plt.subplots(figsize=(12, 4))
            for name in names:
                pts = [r for r in chosen if r["name"] == name]
                ax.plot([r[xkey] for r in pts], [r["rate"] for r in pts], marker="o", label=name)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("tokens/sec")
            ax.legend()
            fig.tight_layout()
            path = csv_path.parent / filename
            fig.savefig(path)
            print(path)

        # Plot - Sweep Runs
        save_lines("B", "b", "batch size", "tokens_per_sec_B.png")
        save_lines("T", "t", "context length", "tokens_per_sec_T.png")


    # Profiling Runs
    if TO_PROFILE:
        if len(PROFILE) > 0:
            by_name = {run["name"]: run for run in RUNS}
            for name in PROFILE:
                run = by_name[name]
                for label, b, t in points(run):
                    profile_dir = csv_path.parent / "profile" / label.replace(" ", "_")
                    cmd = command(run, b, t, csv_path)
                    cmd += [
                        "--profile",
                        "--profile_dir", str(profile_dir),
                        "--profile_wait", str(PROFILE_WAIT),
                        "--profile_warmup", str(PROFILE_WARMUP),
                        "--profile_active", str(PROFILE_ACTIVE),
                        "--profile_traces", str(PROFILE_TRACES).lower(),
                    ]
                    print("profile", label)
                    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()