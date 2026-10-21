import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

# ========================================================
# Global settings
# ========================================================
ROOT = Path(__file__).resolve().parents[1]
CONFIG = "config.local"
WARMUP = 5
BENCHMARK = 50
B_SWEEP = [2, 4, 8]
T_SWEEP = [32, 64, 128]

# ========================================================
# Global Setting: Profiling
# ========================================================
# CUDA memory plots. Leave empty on CPU.
PROFILE = []
# PROFILE = ["baseline", "checkpoint", "accum"]
PROFILE_WAIT = 1
PROFILE_WARMUP = 1
PROFILE_ACTIVE = 4
PROFILE_REPEAT = 1

# ========================================================
# Settings: CPU
# ========================================================
CPU = {
    "ctx_len": 32,
    "use_flash": True,
    "use_checkpoint": False,
    "batch_size": 8,
    "grad_accum": 1,
    "precision": "fp32",
    "compile_model": False,
    "optimizer": "adamw",
    "fused_optimizer": False,
}

# ========================================================
# RUNS
# ========================================================
RUNS = [
    {"name": "baseline", **CPU},
    {"name": "checkpoint", **CPU, "use_checkpoint": True},
    {"name": "accum", **CPU, "batch_size": 2, "grad_accum": 4},
    {"name": "compile", **CPU, "compile_model": True},
    {"name": "manual_attn", **CPU, "use_flash": False},
    {"name": "compile+checkpoint", **CPU, "compile_model": True, "use_checkpoint": True},
    {"name": "checkpoint+manual_attn", **CPU, "use_checkpoint": True, "use_flash": False},

    # To sweep B or T, add a sweep key as shown below:
    {"name": "baseline_B", **CPU, "sweep": "B"},
    {"name": "baseline_T", **CPU, "sweep": "T"},
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
    fig, ax = plt.subplots(figsize=(8, 4))
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
        fig, ax = plt.subplots(figsize=(8, 4))
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
                "--profile_repeat", str(PROFILE_REPEAT),
            ]
            print("profile", label)
            subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()