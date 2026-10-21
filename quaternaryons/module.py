"""
quaternaryons.module — QuATernaryonLinear PyTorch Module
Drop-in replacement for nn.Linear in Hugging Face Transformers (MiniCPM, SmolLM, LLaMA).
Executes dual-sided sub-2-bit inference via:
    y = (x R^T) (beta + repeat(alpha, g) * T + U V^T)^T + bias
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from .butterfly import QuATernaryonRotorNetwork
from .quant import compute_bpw, fit_low_rank_residual, quantize_ternary_asymmetric, ternary_entropy


class QuATernaryonLinear(nn.Module):
    """
    Sub-2-bit Linear layer with 4D Clifford Rotor Butterfly and low-rank SVD correction.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = 32,
        rank: int = 8,
        bias: bool = False,
        trainable_quant: bool = False
    ):
        super().__init__()
        assert in_features % 64 == 0, f"in_features ({in_features}) must be multiple of 64"
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.rank = rank
        self.n_groups = in_features // group_size

        # 4D Quaternion Rotor Butterfly network on input dimension
        self.rotor = QuATernaryonRotorNetwork(in_features)

        # Frozen integer ternary codes T in {-1, 0, +1}
        self.register_buffer("T", torch.zeros(out_features, in_features, dtype=torch.float32))

        # Quantization parameters (continuous parameters can be trained in QAT polish)
        alpha = torch.ones(out_features, self.n_groups, dtype=torch.float32)
        beta = torch.zeros(out_features, 1, dtype=torch.float32)
        U = torch.zeros(out_features, rank, dtype=torch.float32)
        V = torch.zeros(in_features, rank, dtype=torch.float32)

        if trainable_quant:
            self.alpha = nn.Parameter(alpha)
            self.beta = nn.Parameter(beta)
            self.U = nn.Parameter(U)
            self.V = nn.Parameter(V)
        else:
            self.register_buffer("alpha", alpha)
            self.register_buffer("beta", beta)
            self.register_buffer("U", U)
            self.register_buffer("V", V)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter("bias", None)

        self.bpw = 2.0  # Default placeholder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward evaluation:
            1. xr = x @ R^T (via O(d log d) quaternion butterfly)
            2. Wq = beta + repeat(alpha, g) * T + U V^T (in rotated basis)
            3. y  = xr @ Wq^T + bias
        """
        # 1. Rotate activations via quaternion butterfly
        xr = self.rotor(x)

        # 2. Dequantize weight in rotated space
        alpha_rep = torch.repeat_interleave(self.alpha, self.group_size, dim=-1)
        Wq = self.beta + alpha_rep * self.T
        if self.rank > 0:
            Wq = Wq + self.U @ self.V.T

        # 3. Linear projection
        y = F.linear(xr, Wq.to(xr.dtype), self.bias.to(xr.dtype) if self.bias is not None else None)
        return y

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        group_size: int = 32,
        rank: int = 8,
        H: Optional[torch.Tensor] = None,
        trainable_quant: bool = False
    ) -> "QuATernaryonLinear":
        """
        Constructs and quantizes a QuATernaryonLinear layer from an existing nn.Linear module.
        """
        in_features = linear.in_features
        out_features = linear.out_features
        has_bias = linear.bias is not None

        mod = cls(
            in_features=in_features,
            out_features=out_features,
            group_size=group_size,
            rank=rank,
            bias=has_bias,
            trainable_quant=trainable_quant
        ).to(linear.weight.device)

        if has_bias:
            mod.bias.data.copy_(linear.bias.data)

        # 1. Extract full orthogonal matrix R from the quaternion butterfly network
        W_fp = linear.weight.data.float()  # (out_features, in_features)
        R = mod.rotor.get_full_matrix().to(W_fp.device)  # (in_features, in_features)

        # 2. Rotate weight into isoclinic Clifford frame: W_rot = W @ R^T
        W_rot = W_fp @ R.T

        # 3. Asymmetric ternary quantization with per-row offset beta
        beta, alpha, T, W_base = quantize_ternary_asymmetric(W_rot, group_size=group_size)

        # 4. Low-rank SVD correction on residual in rotated space
        E_rot = W_rot - W_base
        H_rot = R @ H @ R.T if H is not None else None
        U, V = fit_low_rank_residual(E_rot, rank=rank, H=H_rot)

        # 5. Populate buffers / parameters
        mod.T.copy_(T.float())
        if trainable_quant:
            mod.alpha.data.copy_(alpha)
            mod.beta.data.copy_(beta)
            mod.U.data.copy_(U)
            mod.V.data.copy_(V)
        else:
            mod.alpha.copy_(alpha)
            mod.beta.copy_(beta)
            mod.U.copy_(U)
            mod.V.copy_(V)

        # 6. Compute exact honest bit-rate
        ent = ternary_entropy(T)
        mod.bpw = compute_bpw(out_features, in_features, group_size, rank, ent)

        return mod
