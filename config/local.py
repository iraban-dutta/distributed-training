from config.constants import VOCAB_SIZE
from src.model.llm_config import LLMConfig
from src.training.trainer import LLMTrainerConfig


# Short run for reading the training loop on CPU.
SEED = 42
B = 8
STEPS = 20
DEVICE = "cpu"


llm_config = LLMConfig(
    vocab_size=VOCAB_SIZE,
    ctx_len=32,
    d_model=64,
    n_layer=2,
    ff_ratio=4,
    dropout=0.0,
    eps=1e-5,
    bias=False,
    n_heads=4,
    n_groups=2,
    use_flash=True,
    use_moe=False,
)


trainer_config = LLMTrainerConfig(
    num_steps=STEPS,
    batch_size=B,
    learning_rate=6e-4,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.95,
    use_lr_scheduler=True,
    warmup_steps=5,
    min_lr=0.1 * 6e-4,
    grad_clip=1.0,
    log_interval=1,
    moe_log_interval=100,
    eval_interval=5,
    eval_steps=2,
    to_save_checkpoint=False,
    checkpoint_interval=5,
    device=DEVICE,
    parallelism="none",
)