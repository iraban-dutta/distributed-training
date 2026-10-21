import numpy as np

from config.constants import TRAIN_BIN, VAL_BIN, TOKEN_DTYPE
from src.model.llm_config import LLMConfig
from src.training.trainer import LLMTrainerConfig, TokenBatchLoader


def make_loaders(trainer_config: LLMTrainerConfig, llm_config: LLMConfig):
    B = trainer_config.batch_size
    # Done to ensure consistency with gradient accumulation
    B_val_test = trainer_config.batch_size * trainer_config.grad_accum_steps
    T = llm_config.ctx_len

    print(f'Create generator for loading train data of shape ({B}, {T})')
    tok_bl = TokenBatchLoader(
        B=B,
        T=T,
        binary_file_path=TRAIN_BIN,
        dtype=TOKEN_DTYPE,
        debug=False,
    )

    print(f'Create generator for loading validation data of shape ({B}, {T})')
    tok_bl_val = TokenBatchLoader(
        # B=B_val_test,
        B=B,
        T=T,
        binary_file_path=VAL_BIN,
        dtype=TOKEN_DTYPE,
        debug=False,
    )

    return tok_bl, tok_bl_val