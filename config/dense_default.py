from config.constants import VOCAB_SIZE
from src.model.llm_config import LLMConfig
from src.training.trainer import LLMTrainerConfig


# ======== SET Global Configs ========
SEED = 42
B = 8
STEPS = 20000
DEVICE = 'auto'


# ======== DEFINE Model Config ========
# Fixed architecture: GQA + RoPE + LayerNorm + dense MLP.
llm_config = LLMConfig(
    vocab_size=VOCAB_SIZE,
    ctx_len=128,
    d_model=512,
    n_layer=8,
    ff_ratio=4,
    dropout=0.0,
    eps=1e-5,
    bias=False,
    n_heads=8,
    n_groups=4,
    use_flash=True,
    use_moe=False,
)


# ======== DEFINE Trainer Config ========
trainer_config = LLMTrainerConfig(
    num_steps=STEPS,
    batch_size=B,
    learning_rate=6e-4,
    weight_decay=0.01,
    beta1=0.9,
    beta2=0.95,
    use_lr_scheduler=True,
    warmup_steps=15,
    min_lr=0.1*6e-4,
    grad_clip=1.0,
    log_interval=10,
    moe_log_interval=100,
    eval_interval=200,
    eval_steps=16,
    to_save_checkpoint=True,
    checkpoint_interval=1000,
    device=DEVICE,
)
