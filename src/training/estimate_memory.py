import torch
import torch.nn as nn
from torch.optim import SGD, AdamW


def parameter_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def gradient_bytes(model: nn.Module) -> int:
    total = 0
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            # Estimate grad memory when PyTorch has not allocated any tensor for the grad
            total += p.numel() * p.element_size()
        else:
            # Estimate grad memory when PyTorch has allocated a tensor for the grad
            total += p.grad.numel() * p.grad.element_size()
    return total


def optimizer_bytes(optimizer) -> int:
    if isinstance(optimizer, AdamW):
        return _adamw_bytes(optimizer)
    if isinstance(optimizer, SGD):
        return _sgd_bytes(optimizer)
    raise NotImplementedError(
        f"{type(optimizer).__name__} memory estimation is not supported"
    )


def _adamw_bytes(optimizer: AdamW) -> int:
    # exp_avg and exp_avg_sq, one each per parameter. The step counter is a scalar.
    total = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            state = optimizer.state.get(p, {})
            exp_avg = state.get("exp_avg")
            exp_avg_sq = state.get("exp_avg_sq")
            if exp_avg is not None and exp_avg_sq is not None:
                # Estimate AdamW state memory when PyTorch has allocated a tensor for it
                total += exp_avg.numel() * exp_avg.element_size()
                total += exp_avg_sq.numel() * exp_avg_sq.element_size()
            else:
                # Estimate AdamW state memory when PyTorch has not allocated a tensor for it
                total += 2 * p.numel() * p.element_size()
    return total


def _sgd_bytes(optimizer: SGD) -> int:
    # A momentum buffer exists only when momentum is not 0.
    total = 0
    for group in optimizer.param_groups:
        if group["momentum"] == 0:
            continue
        for p in group["params"]:
            buf = optimizer.state.get(p, {}).get("momentum_buffer")
            if buf is not None:
                # Estimate momentum buffer memory when PyTorch has allocated a tensor for it
                total += buf.numel() * buf.element_size()
            else:
                # Estimate momentum buffer memory when PyTorch has not allocated a tensor for it
                total += p.numel() * p.element_size()
    return total


def activation_bytes(config, batch_size: int, bytes_per_element: int = 4) -> dict:
    """Bytes of the large tensors backward keeps for one dense step.

    Counted: the logits, one residual tensor per block, the MLP hidden
    tensor per block, and the attention scores when they are materialized.
    use_flash=True uses scaled_dot_product_attention, which does not store
    the T×T score matrix. LayerNorm copies, Q, K, and V are extra copies of
    the residual and are left out of this lower estimate.
    """
    if config.use_moe:
        raise NotImplementedError("activation estimate is for the dense model")

    b, t, d = batch_size, config.ctx_len, config.d_model

    # ================================
    # Logits: 
    # ================================
    logits = b * t * config.vocab_size

    # ================================
    # Decoder block: residual + mlp + attn_scores
    # ================================
    # Residual: 
    residual = b * t * d
    # MLP: 
    mlp = b * t * config.ff_ratio * d
    # Attention scores:
    attn_scores = 0 if config.use_flash else b * config.n_heads * t * t
    # Final decoder block:
    decoder_block = config.n_layer * (residual + mlp + attn_scores)

    return {
        "logits_bytes": logits * bytes_per_element,
        "block_bytes": decoder_block * bytes_per_element,
        "total_bytes": (logits + decoder_block) * bytes_per_element,
    }


def device_memory(device: torch.device) -> dict | None:
    """
    Allocator statistics after the caller has finished the train step.
    - CUDA reports current and peak bytes. 
    - MPS reports only the bytes still allocated, not a peak. 
    - CPU has no allocator statistics.
    """
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return {
            "allocated": torch.cuda.memory_allocated(device),
            "reserved": torch.cuda.memory_reserved(device),
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
        }
    if device.type == "mps":
        torch.mps.synchronize()
        return {
            "allocated": torch.mps.current_allocated_memory(),
            "reserved": None,
            "peak_allocated": None,
            "peak_reserved": None,
        }
    return None