import torch
from einops import einsum, repeat


class RoPE:
    def __init__(self, head_dim:int, base:int=10000):

        # Validate that head_dim should be an even number
        assert head_dim % 2 == 0, (
            f'Head dimension should be even, got {head_dim}'
        )

        self.head_dim = head_dim
        self.base = base

    def _get_m_thetas(self, x:torch.Tensor, seq_len:int, seq_offset:int) -> torch.Tensor:

        # Step1: Build thetas | Shape: (d/2, )
        # 1/theta_i = 10000**(2(i-1)/d) for all i in [1, 2, 3, ...d/2]

        theta_denom = self.base**(torch.arange(0, self.head_dim, 2, device=x.device)/self.head_dim)
        theta = 1/theta_denom

        # Step2: Build ms  | Shape: (m, )
        m = torch.arange(seq_offset+1, seq_offset+1+seq_len, device=x.device)

        # Step3: Take outer product | Shape: (m, d/2)
        # (can use torch.outer as well) 
        m_thetas = einsum(m, theta, "t, d -> t d")

        # Replicate m_thetas across dim=-1 | Shape: (m, d)
        m_thetas = repeat(m_thetas, "t d -> t (d n)", n=2)

        return m_thetas


    def _get_x_flipped(self, x:torch.Tensor) -> torch.Tensor:

        # Step1: Reshape the x to the below form (x -> x_flipped)
        # Basic flipping idea
        # x =         [[x11 x12 x13 x14], 
        #              [x21 x22 x23 x24]]
        # x_flipped = [[-x12 x11 -x14 x13], 
        #              [-x22 x21 -x24 x23]]

        # Actual x will be in higher dim (since it will have B:batch and H:head dimension)
        # x.shape: (B, H, m, d)
        B, H, m, _ = x.shape

        # 1.1: Copy and reshape
        x_copy = x.view(B, H, m, -1, 2)     # Shape: (B, H, m, d/2, 2)

        # 1.2: Pluck out the even and odd dims
        x_copy1 = x_copy[:, :, :, :, 0]          # Shape: (B, H, m, d/2) | x_copy1=x_copy[..., 0] (Clean way to write)
        x_copy2 = x_copy[:, :, :, :, 1]          # Shape: (B, H, m, d/2) | x_copy2=x_copy[..., 1] (Clean way to write)

        # 1.3: Stack up the plucked out tensors in correct order with negative sign
        x_flipped = torch.stack(
            [-x_copy2, x_copy1], 
            dim=-1
        )                                        # Shape: (B, H, m, d/2, 2)

        # 1.4 # Reshape back to original shape as x
        x_flipped = x_flipped.view(B, H, m, -1)  # Shape: (B, H, m, d)

        return x_flipped


    def apply_rope(self, x:torch.Tensor, seq_offset:int=0) -> torch.Tensor:
        # x.shape: B, H, m, d
        _, _, m, _ = x.shape

        # 1: Get m_thetas 
        # m_thetas.shape : (m, d)
        m_thetas = self._get_m_thetas(x, seq_len=m, seq_offset=seq_offset)

        # 2: Create cos(m_thetas) and sin(m_thetas) 
        # Shape          : (m, d)
        m_theta_cos = torch.cos(m_thetas)
        m_theta_sin = torch.sin(m_thetas)

        # 3. Get x_flipped 
        # x.shape        : (B, H, m, d)
        # x_flipped.shape: (B, H, m, d)
        x_flipped = self._get_x_flipped(x)

        # 4: Final additions to create rotated vector: Element wise product (Implicit: Broadcasting happens here)
        # x_rot.shape    : (B, H, m, d)
        x_rot = (x*m_theta_cos) + (x_flipped*m_theta_sin)

        return x_rot
