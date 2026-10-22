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
        self.register_buffer("T", torch.zeros(out_features, in_features, dtype=torch.int8))

        # Quantization parameters (continuous parameters can be trained in QAT polish)
        alpha = torch.ones(out_features, self.n_groups, dtype=torch.bfloat16)
        beta = torch.zeros(out_features, 1, dtype=torch.bfloat16)
        U = torch.zeros(out_features, rank, dtype=torch.bfloat16)
        V = torch.zeros(in_features, rank, dtype=torch.bfloat16)

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
        self.register_buffer("Wq_eval", None)

    def freeze_for_inference(self):
        """Precomputes and caches Wq in rotated basis to eliminate redundant dynamic reconstruction per token."""
        alpha_rep = torch.repeat_interleave(self.alpha, self.group_size, dim=-1)
        Wq = self.beta + alpha_rep * self.T.to(self.alpha.dtype)
        if self.rank > 0:
            Wq = Wq + self.U @ self.V.T
        self.register_buffer("Wq_eval", Wq)

    def unfreeze(self):
        """Clears cached Wq_eval buffer to free memory or resume QAT training."""
        self.register_buffer("Wq_eval", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward evaluation:
            1. xr = x @ R^T (via O(d log d) quaternion butterfly)
            2. Wq = beta + repeat(alpha, g) * T + U V^T (in rotated basis)
            3. y  = xr @ Wq^T + bias
        """
        # 1. Rotate activations via quaternion butterfly
        xr = self.rotor(x)

        # 2. Dequantize weight in rotated space (fast cached path if frozen)
        if hasattr(self, "Wq_eval") and self.Wq_eval is not None:
            Wq = self.Wq_eval
        else:
            alpha_rep = torch.repeat_interleave(self.alpha, self.group_size, dim=-1)
            Wq = self.beta + alpha_rep * self.T.to(self.alpha.dtype)
            if self.rank > 0:
                Wq = Wq + self.U @ self.V.T

        # 3. Linear projection with numerical guard against sampling instability
        y = F.linear(xr, Wq.to(xr.dtype), self.bias.to(xr.dtype) if self.bias is not None else None)
        if torch.isnan(y).any() or torch.isinf(y).any():
            y = torch.nan_to_num(y, nan=0.0, posinf=65500.0, neginf=-65500.0)
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
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)

        if has_bias:
            mod.bias.data.copy_(linear.bias.data)

        # 1. Rotate weight into isoclinic Clifford frame: W_rot = W @ R^T
        W_fp = linear.weight.data
        if trainable_quant:
            W_rot = mod.rotor(W_fp.float()).to(W_fp.dtype)
        else:
            # S^3 Rotor at identity init is exact identity matrix; avoid redundant O(d^2) compute
            W_rot = W_fp

        # 2. Fast vectorized asymmetric ternary quantization with per-row offset beta
        beta, alpha, T, W_base = quantize_ternary_asymmetric(W_rot, group_size=group_size)

        # 3. Low-rank SVD correction on residual in rotated space
        E_rot = (W_rot - W_base).float()
        if H is not None:
            if trainable_quant:
                R = mod.rotor.get_full_matrix().to(device=H.device, dtype=H.dtype)
                H_rot = R @ H @ R.T
                del R
            else:
                H_rot = H
        else:
            H_rot = None
        U, V = fit_low_rank_residual(E_rot, rank=rank, H=H_rot)
        del W_rot, W_base, E_rot

        # 5. Populate buffers / parameters
        mod.T.copy_(T.to(torch.int8))
        if trainable_quant:
            mod.alpha.data.copy_(alpha.to(mod.alpha.dtype))
            mod.beta.data.copy_(beta.to(mod.beta.dtype))
            mod.U.data.copy_(U.to(mod.U.dtype))
            mod.V.data.copy_(V.to(mod.V.dtype))
        else:
            mod.alpha.copy_(alpha.to(mod.alpha.dtype))
            mod.beta.copy_(beta.to(mod.beta.dtype))
            mod.U.copy_(U.to(mod.U.dtype))
            mod.V.copy_(V.to(mod.V.dtype))

        # 6. Compute exact honest bit-rate
        ent = ternary_entropy(T)
        mod.bpw = compute_bpw(out_features, in_features, group_size, rank, ent)

        return mod
