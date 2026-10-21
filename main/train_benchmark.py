import argparse
import importlib
import time
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from src.model.llm import LLM
from src.training.trainer import LLMTrainer
from src.training.benchmark import append_csv, benchmark_steps, summarize  
from main.run_utils import make_loaders

# ==============================================================================
# Benchmark-only settings (not part of the long training config)
# ==============================================================================
WARMUP_STEPS = 50
BENCHMARK_STEPS = 100

# ==============================================================================
# Sync device
# ==============================================================================
def sync_device(device: torch.device) -> None:
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elif device.type == 'mps':
        torch.mps.synchronize()

# ==============================================================================
# CLI — same config module as train.py
# ==============================================================================

# Helper function to convert string to boolean
def str2bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true or false, got {value}")

# Parser for CLI arguments
parser = argparse.ArgumentParser()

# Config Module Config
parser.add_argument(
    '--config',
    default='config.dense_default',
    help='Config module, e.g. config.dense_default',
)
# Benchmark CSV File Config
parser.add_argument(
    "--benchmark_csv",
    type=str,
    default="runs/benchmark/benchmarks.csv",
    help="CSV file for the throughput benchmark",
)
# Profile Directory Config
parser.add_argument(
    "--profile_dir",
    type=str,
    default="runs/profile/",
    help="Directory for the trace, operators.txt, and memory.html",
)

# ================================================
# Model Configs
# ================================================

# Context Length Config
parser.add_argument(
    "--ctx_len", 
    type=int, 
    default=None,
    help='Override context length for TPUT benchmark'
)
# Activation Checkpointing Config
parser.add_argument(
    "--use_activation_checkpoint", 
    type=str2bool, 
    default=None,
    help='Override use activation checkpoint for TPUT benchmark'
)
# Optimized Attention Kernel Config
parser.add_argument(
    "--use_flash", 
    type=str2bool, 
    default=None,
    help='Override use optimized attention kernel for TPUT benchmark'
)


# ================================================
# Trainer Configs
# ================================================

# Batch Size Config
parser.add_argument(
    '--batch_size', 
    type=int, 
    default=None, 
    help='Override batch size for TPUT benchmark'
)
# Grad Accumulation Config
parser.add_argument(
    "--grad_accum", 
    type=int, 
    default=None,
    help='Override grad accumulation steps for TPUT benchmark'
)
# Optimizer Configs
parser.add_argument(
    "--optimizer", 
    type=str, 
    default=None,
    help='Override optimizer for TPUT benchmark'
)
parser.add_argument(
    "--fused_optimizer", 
    type=str2bool, 
    default=None,
    help='Override fused optimizer for TPUT benchmark'
)
# Precision Config
parser.add_argument(
    "--precision", 
    type=str, 
    default=None,
    help='Override precision for TPUT benchmark'
)
# torch.compile Config
parser.add_argument(
    "--compile_model", 
    type=str2bool, 
    default=None,
    help='Override torch.compile model for TPUT benchmark'
)


# ================================================
# Benchmark Configs
# ================================================

# Warmup Steps Config
parser.add_argument(
    '--warmup', 
    type=int, 
    default=None, 
    help='Override warmup steps for TPUT benchmark'
)
# Benchmark Steps Config
parser.add_argument(
    '--benchmark', 
    type=int, 
    default=None, 
    help='Override benchmark steps for TPUT benchmark'
)

# ================================================
# Profiler Configs
# ================================================

# Profile Config
parser.add_argument(
    "--profile", 
    action="store_true",
    help='Profile the training step using torch.profiler'
)
# Profile Wait Time Config
parser.add_argument(
    "--profile_wait", 
    type=int, 
    default=None,
    help='Override wait time for profile'
)
# Profile Warmup Time Config
parser.add_argument(
    "--profile_warmup", 
    type=int, 
    default=None,
    help='Override warmup time for profile'
)
# Profile Active Time Config
parser.add_argument(
    "--profile_active", 
    type=int, 
    default=None,
    help='Override active time for profile'
)
# Profile Repeats Config
parser.add_argument(
    "--profile_repeat", 
    type=int, 
    default=None,
    help='Override repeats for profile'
)

# ================================================
# Parse Arguments
# ================================================
args = parser.parse_args()

# Import config from provided path (args.config: config.py module path)
cfg = importlib.import_module(args.config)

# Set seed
torch.manual_seed(cfg.SEED)

# Define model and trainer configs
llm_config = cfg.llm_config
trainer_config = cfg.trainer_config

# Apply CLI overrides on top of trainer_config (config file or checkpoint)

# ================================================
# Model Config Overrides
# ================================================
# Model Config Overrides
if args.ctx_len is not None:
    llm_config.ctx_len = args.ctx_len
    print(f'Override context length for TPUT benchmark: {args.ctx_len}')
    print('-' * 50)
if args.use_activation_checkpoint is not None:
    llm_config.use_checkpoint = args.use_activation_checkpoint
    print(f'Override use activation checkpoint for TPUT benchmark: {args.use_activation_checkpoint}')
    print('-' * 50)
if args.use_flash is not None:
    llm_config.use_flash = args.use_flash
    print(f'Override use optimized attention kernel for TPUT benchmark: {args.use_flash}')
    print('-' * 50)

# ================================================
# Trainer Config Overrides
# ================================================
if args.batch_size is not None:
    trainer_config.batch_size = args.batch_size
    print(f'Override batch size for TPUT benchmark: {args.batch_size}')
    print('-' * 50)
if args.grad_accum is not None:
    trainer_config.grad_accum_steps = args.grad_accum
    print(f'Override grad accumulation steps for TPUT benchmark: {args.grad_accum}')
    print('-' * 50)
if args.optimizer is not None:
    trainer_config.optimizer_name = args.optimizer
    print(f'Override optimizer for TPUT benchmark: {args.optimizer}')
    print('-' * 50)
if args.fused_optimizer is not None:
    trainer_config.fused_optimizer = args.fused_optimizer
    print(f'Override fused optimizer for TPUT benchmark: {args.fused_optimizer}')
    print('-' * 50)
if args.precision is not None:
    trainer_config.precision = args.precision
    print(f'Override precision for TPUT benchmark: {args.precision}')
    print('-' * 50)
if args.compile_model is not None:
    trainer_config.compile_model = args.compile_model
    print(f'Override torch.compile model for TPUT benchmark: {args.compile_model}')
    print('-' * 50)


# ================================================
# Benchmark Config Overrides
# ================================================
if args.warmup is not None:
    WARMUP_STEPS = args.warmup
    print(f'Override warmup steps for TPUT benchmark: {args.warmup}')
    print('-' * 50)

if args.benchmark is not None:
    BENCHMARK_STEPS = args.benchmark
    print(f'Override benchmark steps for TPUT benchmark: {args.benchmark}')
    print('-' * 50)


# ==============================================================================
# Training Setup
# ==============================================================================

# Define batch size and context length
B = trainer_config.batch_size
T = llm_config.ctx_len

print(f'Benchmark with config: {args.config}')
print(f'Batch size B={B}, context T={T}')
print(llm_config)
print('-' * 50)
print(trainer_config)
print('-' * 50)

# Create data loaders
tok_bl, _ = make_loaders(trainer_config, llm_config)

# Create model
model = LLM(config=llm_config)

# Create trainer
trainer = LLMTrainer(
    config=trainer_config,
    model=model,
    train_loader=tok_bl,
    val_loader=None,
)

# ==============================================================================
# Profile the training step
# ==============================================================================
if args.profile:
    print('Starting Profile')
    print('-' * 50)

    # Import the profiler
    from src.training.profiler import profile_train_step

    # Create run directory
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = str(Path(args.profile_dir) / stamp)

    # Override the settings if provided
    wait, warmup, active, repeats = 5, 5, 40, 2
    if args.profile_wait is not None:
        wait = args.profile_wait
    if args.profile_warmup is not None:
        warmup = args.profile_warmup
    if args.profile_active is not None:
        active = args.profile_active
    if args.profile_repeat is not None:
        repeats = args.profile_repeat

    # Call the profiler
    prof, trace_path = profile_train_step(trainer, run_dir, wait, warmup, active, repeats)
    
    # Print the top 15 operators by time
    sort_by = "self_cuda_time_total" if trainer.device.type == "cuda" else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort_by, row_limit=15)
    print(table)
    print('-' * 50)
    # Save the table to a file
    Path(run_dir, "operators.txt").write_text(table + "\n")

    # Print the trace file: Navigate to perfetto.dev to view the trace
    print(f"Trace file: {trace_path}")
    print('-' * 50)

    # Dump the memory timeline
    if trainer.device.type == "cuda":
        index = trainer.device.index
        if index is None:
            index = torch.cuda.current_device()
        memory_device = f"cuda:{index}"
        memory_path = Path(run_dir, "memory.html")
        prof.export_memory_timeline(str(memory_path), device=memory_device)
        print(f"Memory plot: {memory_path} ({memory_device})")

    print('Finished Profile')

    # Exit the program
    raise SystemExit(0)

# ==============================================================================
# Benchmark
# ==============================================================================
print('Starting Benchmark')
print('-' * 50)
step_s = benchmark_steps(trainer, WARMUP_STEPS, BENCHMARK_STEPS)
stats = summarize(step_s, B * trainer.config.grad_accum_steps, T)
print(
    f"median {stats['median_step_ms']:.2f} ms, "
    f"mean {stats['mean_step_ms']:.2f} ms, "
    f"{stats['tokens_per_sec']:.2f} tok/s"
)
row = {
    "config": args.config,
    "device": trainer.device.type,                        # Device Type
    "precision": trainer.config.precision,                # Mixed Precision
    "compile_model": trainer.config.compile_model,        # torch.compile
    "grad_accum_steps": trainer.config.grad_accum_steps,  # Grad Accumulation
    "use_checkpoint": model.config.use_checkpoint,        # Activation Checkpointing
    "optimizer_name": trainer.config.optimizer_name,      # Optimizer Name
    "fused_optimizer": trainer.config.fused_optimizer,    # Fused Optimizer
    "use_flash": model.config.use_flash,                  # Optimized Attention Kernel
    "batch_size": B,
    "seq_len": T,
    "warmup_steps": WARMUP_STEPS,
    "benchmark_steps": BENCHMARK_STEPS,
    "median_step_ms": round(stats["median_step_ms"], 3),
    "mean_step_ms": round(stats["mean_step_ms"], 3),
    "tokens_per_sec": round(stats["tokens_per_sec"], 3),
    "torch": torch.__version__,
}
append_csv(args.benchmark_csv, row)


print(f'Finished Benchmark, Mean Tokens/Sec: {round(stats["tokens_per_sec"], 3):.2f}')
print('-' * 50)


# ================================  OLD CODE  ====================================

# # ==============================================================================
# # Warmup
# # ==============================================================================
# print('Starting Warmup')
# print('-' * 50)
# for i in range(WARMUP_STEPS):
#     print(f'Running WARMUP step: {i + 1}')
#     _, _ = trainer.train_step()

# # Synchronize device
# sync_device(trainer.device)

# print('Finished Warmup')
# print('-' * 50)


# # ==============================================================================
# # Benchmark
# # ==============================================================================
# print('Starting Benchmark')
# print('-' * 50)
# tokens_per_sec_hist = []

# for i in range(BENCHMARK_STEPS):

#     # Synchronize device
#     sync_device(trainer.device)
#     start = time.perf_counter()

#     train_loss, grad_norm = trainer.train_step()

#     # Synchronize device
#     sync_device(trainer.device)
#     end = time.perf_counter()
    
#     bt = end - start
#     tokens_per_sec = (B * T) / bt
#     tokens_per_sec_hist.append(tokens_per_sec)

#     print(
#         f"Step: {i + 1}/{BENCHMARK_STEPS}, "
#         f"GradNorm: {grad_norm:.2f}, Loss: {train_loss:.4f}, "
#         f"Batch Time: {bt * 1000:.2f}ms, Tokens/Sec: {tokens_per_sec:.2f}"
#     )

#     if model.config.use_moe and (i % 10 == 0):
#         trainer._log_moe_stats()

# print(f'Finished Benchmark, Mean Tokens/Sec: {np.mean(tokens_per_sec_hist):.2f}')
# print('-' * 50)