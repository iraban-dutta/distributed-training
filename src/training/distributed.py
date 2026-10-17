import torch.distributed as dist
import torch.nn as nn


def world_size() -> int:
    """Processes in this launch. 1 when the run was not started with torchrun."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def prepare(model: nn.Module, config, device) -> nn.Module:
    """Move the model for this run. Wrapping comes later, in this function."""
    kind = config.parallelism
    size = world_size()

    if kind == "none":
        if size != 1:
            raise RuntimeError(
                f"parallelism='none' but {size} processes are running"
            )
        return model.to(device)

    if size == 1:
        raise RuntimeError(
            f"parallelism={kind!r} needs more than one process"
        )

    if kind == "tp" and model.config.n_heads % size != 0:
        raise ValueError(f"{size} processes must divide n_heads ({model.config.n_heads})")
    elif kind == "pp" and model.config.n_layer % size != 0:
        raise ValueError(f"{size} processes must divide n_layer ({model.config.n_layer})")
    elif kind == "ep":
        if not model.config.use_moe:
            raise ValueError("ep requires use_moe=True")
        if model.config.n_experts % size != 0:
            raise ValueError(f"{size} processes must divide n_experts ({model.config.n_experts})")
    elif kind not in {"ddp", "fsdp"}:
        raise ValueError(f"unknown parallelism {kind!r}")

    raise NotImplementedError(
        f"parallelism={kind!r} resolved to {size} processes, and is not implemented yet"
    )