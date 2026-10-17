import argparse
import importlib
import time
import torch
import numpy as np
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
parser = argparse.ArgumentParser()
parser.add_argument(
    '--config',
    default='config.dense_default',
    help='Config module, e.g. config.dense_default',
)
parser.add_argument(
    '--batch-size', 
    type=int, 
    default=None, 
    help='Override batch size for TPUT benchmark'
)
parser.add_argument(
    '--warmup', 
    type=int, 
    default=None, 
    help='Override warmup steps for TPUT benchmark'
)
parser.add_argument(
    '--benchmark', 
    type=int, 
    default=None, 
    help='Override benchmark steps for TPUT benchmark'
)
args = parser.parse_args()

# Import config from provided path (args.config: config.py module path)
cfg = importlib.import_module(args.config)

# Set seed
torch.manual_seed(cfg.SEED)

# Define model and trainer configs
llm_config = cfg.llm_config
trainer_config = cfg.trainer_config

# Apply CLI overrides on top of trainer_config (config file or checkpoint)
if args.batch_size is not None:
    trainer_config.batch_size = args.batch_size
    print(f'Override batch size for TPUT benchmark: {args.batch_size}')
    print('-' * 50)

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
# Benchmark
# ==============================================================================
print('Starting Benchmark')
print('-' * 50)
step_s = benchmark_steps(trainer, WARMUP_STEPS, BENCHMARK_STEPS)
stats = summarize(step_s, B, T)
print(
    f"median {stats['median_step_ms']:.2f} ms, "
    f"mean {stats['mean_step_ms']:.2f} ms, "
    f"{stats['tokens_per_sec']:.2f} tok/s"
)
row = {
    "config": args.config,
    "device": trainer.device.type,
    "precision": "fp32",
    "batch_size": B,
    "seq_len": T,
    "warmup_steps": WARMUP_STEPS,
    "benchmark_steps": BENCHMARK_STEPS,
    "median_step_ms": round(stats["median_step_ms"], 3),
    "mean_step_ms": round(stats["mean_step_ms"], 3),
    "tokens_per_sec": round(stats["tokens_per_sec"], 3),
    "torch": torch.__version__,
}
append_csv("results/benchmarks.csv", row)

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