from dataclasses import dataclass


@dataclass
class LLMConfig:

    # ================================================================
    # Model
    # ================================================================
    vocab_size: int = 50257
    ctx_len: int = 8
    d_model: int = 32
    n_layer: int = 2
    ff_ratio: int = 4
    dropout: float = 0.0
    eps: float = 1e-5
    bias: bool = True

    # ================================================================
    # Attention is always GQA. Positions are always RoPE.
    # Normalization is always LayerNorm.
    # ================================================================
    n_heads: int = 4
    n_groups: int = 2
    use_flash: bool = False

    # ================================================================
    # Feed-forward: dense MLP, or sparse MoE when use_moe=True
    # ================================================================
    use_moe: bool = False

    # ================================================================
    # MoE
    # ================================================================

    # ------------------------------------------------
    # Routing
    # ------------------------------------------------
    n_experts: int = 2
    topk: int = 1
    capacity_factor: float = 1.0
    use_vectorized_dispatch: bool = True

    # Noisy Router
    noisy_router: bool = False
    router_noise_std: float = 0.0

    # ------------------------------------------------
    # Experts
    # ------------------------------------------------
    n_shared_experts: int = 0

    # ------------------------------------------------
    # Auxiliary Losses
    # ------------------------------------------------
    scale_aux_loss_expert_imp: float = 0.0
    scale_aux_loss_load_balance: float = 0.0
    aux_loss_free_load_balance: bool = False
    aux_loss_free_load_balance_bias_update: float = 0.0



    # ================================================================
    # Validation
    # ================================================================
    def __post_init__(self):
        self.validate()

    def validate(self) -> None:

        # ============================================================
        # Basic Configs
        # ============================================================
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be > 0.")

        if self.ctx_len <= 0:
            raise ValueError("ctx_len must be > 0.")

        if self.d_model <= 0:
            raise ValueError("d_model must be > 0.")

        if self.d_model % 2 != 0:
            raise ValueError("d_model must be even.")

        if self.n_layer <= 0:
            raise ValueError("n_layer must be > 0.")

        if self.ff_ratio <= 0:
            raise ValueError("ff_ratio must be > 0.")

        if self.dropout < 0 or self.dropout > 1:
            raise ValueError("dropout must be within [0, 1].")

        if self.eps <= 0:
            raise ValueError("eps must be > 0.")

        # ============================================================
        # GQA + RoPE
        # ============================================================
        if self.n_heads <= 0:
            raise ValueError("n_heads must be > 0.")

        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})."
            )

        head_dim = self.d_model // self.n_heads
        if head_dim % 2 != 0:
            raise ValueError(
                "RoPE requires an even head dimension "
                f"(d_model // n_heads = {head_dim})."
            )

        if self.n_groups <= 0:
            raise ValueError("n_groups must be > 0.")

        if self.n_groups > self.n_heads:
            raise ValueError(
                f"n_groups ({self.n_groups}) cannot exceed "
                f"n_heads ({self.n_heads})."
            )

        if self.n_heads % self.n_groups != 0:
            raise ValueError("n_heads must be divisible by n_groups.")

        # ============================================================
        # MoE
        # ============================================================
        self._validate_moe()


    def _validate_moe(self) -> None:

        # ============================================================
        # MoE
        # ============================================================
        if not self.use_moe:
            return

        # ------------------------------------------------------------
        # Routing
        # ------------------------------------------------------------
        if self.n_experts <= 1:
            raise ValueError(
                "MoE requires n_experts > 1."
            )

        if self.topk <= 0:
            raise ValueError(
                "MoE topk must be > 0."
            )

        if self.topk > self.n_experts:
            raise ValueError(
                f"MoE topk ({self.topk}) cannot be greater than "
                f"n_experts ({self.n_experts})."
            )

        if self.capacity_factor <= 0:
            raise ValueError(
                "MoE capacity_factor must be > 0."
            )

        # ------------------------------------------------------------
        # Noisy Router
        # ------------------------------------------------------------
        if self.noisy_router and self.router_noise_std <= 0:
            raise ValueError(
                "router_noise_std must be > 0 when noisy_router=True."
            )

        # ------------------------------------------------------------
        # Auxiliary Losses
        # ------------------------------------------------------------
        if self.scale_aux_loss_expert_imp < 0:
            raise ValueError(
                "scale_aux_loss_expert_imp must be >= 0."
            )

        if self.scale_aux_loss_load_balance < 0:
            raise ValueError(
                "scale_aux_loss_load_balance must be >= 0."
            )

        if self.aux_loss_free_load_balance_bias_update < 0:
            raise ValueError(
                "aux_loss_free_load_balance_bias_update must be >= 0."
            )

        # DeepSeek-style auxiliary-loss-free load balancing
        # is an alternative to the explicit load-balance loss.
        if (
            self.aux_loss_free_load_balance
            and self.scale_aux_loss_load_balance > 0
        ):
            raise ValueError(
                "aux_loss_free_load_balance=True cannot be used "
                "with scale_aux_loss_load_balance > 0."
            )


        # ------------------------------------------------------------
        # Expert Configuration
        # ------------------------------------------------------------
        if self.n_shared_experts < 0:
            raise ValueError(
                "n_shared_experts must be >= 0."
            )
