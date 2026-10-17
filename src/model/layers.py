import torch
import torch.nn as nn
from .llm_config import LLMConfig
from .attention import GroupedQueryAttention
from .normalization import LayerNorm
from .moe import MLP, MoE
from src.inference.cache import KVCache


class Decoder(nn.Module):
    def __init__(self, config:LLMConfig):
        super().__init__()
        # LayerNorm for Attention Block
        self.ln_attn=LayerNorm(config)
        # Grouped-query attention with RoPE
        self.attn=GroupedQueryAttention(config)
        # LayerNorm for MLP Block
        self.ln_mlp=LayerNorm(config)
        # Dense MLP or sparse MoE
        self.mlp=MLP(config) if not config.use_moe else MoE(config)
        
    def forward(self, x:torch.Tensor, cache:KVCache|None=None) -> torch.Tensor:
        # x.shape = (B, T, d_model)

        # PreNorm Architecture with a clean residual stream
        x = x + self.attn(self.ln_attn(x), cache)
        x = x + self.mlp(self.ln_mlp(x))

        return x
