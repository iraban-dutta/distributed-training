import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .llm_config import LLMConfig
from .layers import Decoder
from .normalization import LayerNorm
from src.inference.cache import KVCacheManager

class LLM(nn.Module):
    def __init__(self, config:LLMConfig):
        super().__init__()
        self.config=config

        model_dict = {
            "wte":nn.Embedding(config.vocab_size, config.d_model),
            "drp":nn.Dropout(config.dropout),
            "dec": nn.ModuleList([Decoder(config) for _ in range(config.n_layer)]),
            "ln_final":LayerNorm(config)

        }
        
        self.transformer = nn.ModuleDict(model_dict)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, config.bias)

        # Weight Tying
        self.transformer.wte.weight = self.lm_head.weight
        assert self.transformer.wte.weight.data_ptr() == self.lm_head.weight.data_ptr(), (
            f"Issue in Weight Tying"
        )
        print('Weight Tying Done: B/W initial Embedding Layer and final Linear Layer before Softmax!')

        # Manual init
        self.apply(self._init_weights)

        # Report #params for model
        print(f"Model initialized, number of parameters: {(self._get_num_params()/1e6):.2f}M")

    def _init_weights(self, module):

        std = 0.02
        if isinstance(module, nn.Linear) and hasattr(module, 'RESIDUAL_PATH_SCALE_INIT'):
            std = 1/((2*self.config.n_layer)**0.5)
            torch.nn.init.normal_(tensor=module.weight, mean=0.0, std=std)
        
        # For nn.Linear.weight: keep init scaling - default
        # For nn.Linear.bias  : keep init scaling - as zero
        if isinstance(module, nn.Linear):
            # torch.nn.init.normal_(tensor=module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(tensor=module.bias)

        # For nn.Embedding.weight: keep init scaling - default
        # if isinstance(module, nn.Embedding):
        #     torch.nn.init.normal_(tensor=module.weight, mean=0.0, std=std)



    def _get_num_params(self) -> int:
        # Token embeddings are tied with lm_head, so parameters() counts them once.
        # RoPE has no parameters.
        return sum(p.nelement() for p in self.parameters())


    def forward(self, x:torch.Tensor, y:None|torch.Tensor=None, cache_manager:KVCacheManager|None=None):

        # x.shape = (B, T)
        B, T = x.shape

        # Resolve Device and move input to device
        x = x.to(next(self.parameters()).device)

        # Check for context overflow: T <= ctx_len
        assert T<=self.config.ctx_len, (
                f"Input shape: {tuple(x.shape)}"
                f"Sequence length in x={T} should be lesser than max context length of model {self.config.ctx_len}"
        )

        # Forward Pass1: Token Embeddings
        # Positions are applied inside attention via RoPE.
        x = self.transformer.wte(x)

        # Forward Pass2: Dropout
        x = self.transformer.drp(x)

        # Forward Pass3: Sequence of Decoder Layers (Norm + Attention + Norm + MLP/MoE)
        for i, decoder in enumerate(self.transformer.dec):
            cache = cache_manager[i] if cache_manager is not None else None
            if self.training and self.config.use_checkpoint:
                x = checkpoint(decoder, x, cache, use_reentrant=False)
            else:
                x=decoder(x, cache)


        # Forward Pass4: Norm (Before LM Head)
        x = self.transformer.ln_final(x)

        # Forward Pass5: LM head
        logits = self.lm_head(x)

        loss = None
        if y is not None:
            # Language Modelling CCE Loss
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
            # MoE auxiliary losses
            if self.config.use_moe:
                aux_loss = 0.0
                for decoder in self.transformer.dec:
                    aux_loss += decoder.mlp.loss_coeff_var
                    aux_loss += decoder.mlp.loss_load_balance

                loss += aux_loss

        return logits, loss
