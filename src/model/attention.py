import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat
from .llm_config import LLMConfig
from .position_embedding import RoPE
from src.inference.cache import KVCache


def compute_attention(
        q:torch.Tensor, 
        k:torch.Tensor, 
        v:torch.Tensor, 
        use_flash:bool,
        dropout_p:float, 
        is_causal_flag:bool, 
        head_dim:int,
        dropout_attn:nn.Module,
        device=torch.device
    ) -> torch.Tensor:

    if use_flash:
        # PyTorch SDPA - launches optimized kernels (Flash Attention on CUDA when available)
        y = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None, 
                dropout_p=dropout_p, 
                is_causal=is_causal_flag
            )
    else:
        # Implement Manual Attention 
        # Attention_p1: Dot Product (Q @ K.T)
        attn_scores = einsum(q, k, "b h i d, b h j d -> b h i j")

        # Attention_p2: Scaling
        attn_scores *= (1/(head_dim)**0.5)

        # Attention_p3: Causal Mask
        if is_causal_flag:
            mask = torch.triu(torch.full_like(attn_scores, fill_value=-torch.inf, device=device), diagonal=1)
            attn_scores += mask

        # Attention_p4: Softmax
        attn_scores = F.softmax(attn_scores, dim=-1)
        attn_scores = dropout_attn(attn_scores)

        # Attention_p5: Dot Product (A @ V)
        y = einsum(attn_scores, v, "b h i j, b h j d -> b h i d")

    return y


class GroupedQueryAttention(nn.Module):
    
    def __init__(self, config:LLMConfig):
        super().__init__()
        self.d_model=config.d_model
        self.n_heads=config.n_heads
        self.n_groups=config.n_groups
        self.head_dim=(config.d_model//config.n_heads)
        # QKV projection layer
        self.proj_qkv = nn.Linear(config.d_model, config.d_model + 2*self.n_groups*self.head_dim)
        # Output projection layer
        self.proj_out = nn.Linear(config.d_model, config.d_model)
        self.proj_out.RESIDUAL_PATH_SCALE_INIT=1
        # Rotary position embeddings are applied to every Q and K.
        self.rope = RoPE(head_dim=self.head_dim)
        # Dropouts
        self.dropout_attn = nn.Dropout(config.dropout)
        self.dropout_out = nn.Dropout(config.dropout)
        self.dropout_p = config.dropout
        # Flash Attn
        self.use_flash = config.use_flash
        if self.use_flash:
            if hasattr(F, 'scaled_dot_product_attention'):
                print("use_flash=True - To use PyTorch's Flash Attention")
            else:
                self.use_flash=False
                print("use_flash=True - PyTorch's Flash Attention needs version >= 2.0 - Fallback to Manual Attention")


    def forward(self, x:torch.Tensor, kv_cache:KVCache|None=None) -> torch.Tensor:
        # x.shape = (B, T, d_model)
        B, T, _ = x.shape

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Get qkv projections in concated form
        # -------------------------------- XX -------------------------------- XX --------------------------------
        qkv = self.proj_qkv(x)

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Split q, k, v
        # -------------------------------- XX -------------------------------- XX --------------------------------
        q, k, v = torch.split(qkv, [self.d_model, self.n_groups*self.head_dim, self.n_groups*self.head_dim], dim=-1)

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Reshape q, k, v by accounting for n_heads, n_groups
        # -------------------------------- XX -------------------------------- XX --------------------------------
        q = rearrange(q, "b s (h d) -> b h s d", h=self.n_heads)
        k = rearrange(k, "b s (g d) -> b g s d", g=self.n_groups)
        v = rearrange(v, "b s (g d) -> b g s d", g=self.n_groups)

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Apply RoPE
        # -------------------------------- XX -------------------------------- XX --------------------------------
        use_kv_cache = kv_cache is not None
        if use_kv_cache:
            # This path used during Inference (Prefill + Decode) with KV Cache enabled
            # Handles sequences during inference time which are longer than ctx_len supported
            # Slides context of the last ctx_len tokens
            # Bake absolute postion into KV cache by rotating by m (pos of current token in seq)
            # This does not matter since during attention calculation, we only get relative position
            q = self.rope.apply_rope(x=q, seq_offset=kv_cache.ntokens_processed)
            k = self.rope.apply_rope(x=k, seq_offset=kv_cache.ntokens_processed)
        else:
            # This path used during Train + Naive Inference
            q = self.rope.apply_rope(x=q)
            k = self.rope.apply_rope(x=k)

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # KV Caching: Activated during Inference if kv_cache is an object of correct class
        # -------------------------------- XX -------------------------------- XX --------------------------------
        if use_kv_cache:

            # Resolve Prefill/Decode stage (this controls behaviour for causal masking)
            is_decode = kv_cache.k_cache is not None

            # Update the KV Cache 
            kv_cache.update_cache(k, v)

            # Prefill Stage: Update the k and v vectors corresponding to all q vectors from prompt tokens, then fetch entire k,v (overrwriting)
            # Decode Stage: Update the k and v vector corresponding to current q (current token), then fetch entire k,v (past + current token)
            # FIX: We are using a KV cache whose size gets allocated at the start of inference, hence very important to slice it upto current token position
            k = kv_cache.k_cache[:, :, :kv_cache.curr_idx] 
            v = kv_cache.v_cache[:, :, :kv_cache.curr_idx]

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Repeat k v to be shared across n heads 
        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Group heads stay contiguous: G0 G0 G1 G1
        k = repeat(k, "b g s d -> b (g n) s d", n=self.n_heads//self.n_groups)
        v = repeat(v, "b g s d -> b (g n) s d", n=self.n_heads//self.n_groups)

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Flag to control causal masking behaviour
        # -------------------------------- XX -------------------------------- XX --------------------------------
        is_causal_flag=True             # Usual Flow (Without Cache/With Cache:Train & Infer-Prefill)
        if use_kv_cache and is_decode:
            is_causal_flag=False        # Overrides flag to false only when Cache is used + Infer-Decode

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Attention
        # -------------------------------- XX -------------------------------- XX --------------------------------
        y = compute_attention(
            q=q, 
            k=k, 
            v=v, 
            use_flash=self.use_flash,
            dropout_p=self.dropout_p if self.training else 0, 
            is_causal_flag=is_causal_flag, 
            head_dim=self.head_dim,
            dropout_attn=self.dropout_attn,
            device=x.device
        )

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Reshape post attention output to original shape of input
        # -------------------------------- XX -------------------------------- XX --------------------------------
        y = rearrange(y, "b h s d -> b s (h d)")

        # -------------------------------- XX -------------------------------- XX --------------------------------
        # Get op projection
        # -------------------------------- XX -------------------------------- XX --------------------------------
        y = self.dropout_out(self.proj_out(y))

        return y
