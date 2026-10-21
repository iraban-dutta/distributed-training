import torch
from pathlib import Path
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
from src.training.trainer import LLMTrainer


def profile_train_step(
    trainer:LLMTrainer, 
    trace_dir:str,
    wait: int=2,
    warmup: int=2,
    active: int=4,
    repeats: int=5,
)-> tuple[profile, str]:
    """
    Profile training steps of the model.
    """

    # Define the events to profile (CPU and CUDA)
    activities = [ProfilerActivity.CPU]
    if trainer.device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    # Create the trace directory
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    # Create the profiler
    prof = profile(
        activities=activities,
        schedule=schedule(wait=wait, warmup=warmup, active=active),
        on_trace_ready=tensorboard_trace_handler(str(trace_dir)),
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=True,
    )

    # Define the total number of steps to profile
    total_steps = (wait + warmup + active) * repeats

    # Run the profiler
    with prof:
        for _ in range(total_steps):
            # Execute a training step
            trainer.train_step()
            # Step the profiler
            prof.step()

    # Get hold of the trace file
    traces = sorted(trace_dir.glob("*.pt.trace.json"))
    if not traces:
        raise FileNotFoundError(f"No traces found in {trace_dir}")

    # Return the profiler and the path to the trace file
    return prof, str(traces[-1])

