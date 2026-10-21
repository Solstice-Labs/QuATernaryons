"""
quaternaryons.butterfly — Multi-Stage 4D Quaternion Rotor Butterfly Network
Implements O(d log4 d) orthogonal transformations parameterized on the S^3 Clifford manifold.
Architected specifically for sub-2-bit LLMs where dimension is a multiple of 64 (e.g. MiniCPM).
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
from .rotor import cayley_matrix_torch, left_matrix_torch, right_matrix_torch


def butterfly_permutation_torch(d: int, stride: int = 4) -> torch.Tensor:
    """
    Deterministic stride permutation matrix for butterfly stage s.
    (P x)[i] = x[perm[i]], where perm[i] = (i % stride) * (d // stride) + (i // stride).
    Dispenses localized activation outliers across distinct 4D quaternion blocks.
    """
    idx = torch.arange(d)
    perm = (idx % stride) * (d // stride) + (idx // stride)
    P = torch.zeros((d, d), dtype=torch.float32)
    P[idx, perm] = 1.0
    return P


class QuaternionButterflyBlock64(nn.Module):
    """
    A 64-dimensional 3-stage Quaternion Butterfly Network (64 = 4^3).
    Has 16 parallel 4x4 quaternion rotor blocks per stage, across log4(64) = 3 stages.
    Total rotor DOFs = 3 stages * 16 blocks * 6 parameters (Cayley A(L)B(R)) = 288 parameters.
    Guaranteed strictly orthogonal: R^T @ R = I_64.
    """

    def __init__(self, use_b_rotors: bool = True):
        super().__init__()
        self.d = 64
        self.num_stages = 3
        self.num_blocks = 16  # 64 / 4
        self.use_b_rotors = use_b_rotors

        # Initialize unit quaternions L (and R) on S^3: shape (3, 16, 4)
        # Identity initialization: q = (1, 0, 0, 0)
        qL_init = torch.zeros(self.num_stages, self.num_blocks, 4)
        qL_init[..., 0] = 1.0
        self.qL = nn.Parameter(qL_init)

        if use_b_rotors:
            qR_init = torch.zeros(self.num_stages, self.num_blocks, 4)
            qR_init[..., 0] = 1.0
            self.qR = nn.Parameter(qR_init)
        else:
            self.register_parameter("qR", None)

        # Precompute permutation matrices for the 3 stages: strides 1, 4, 16
        strides = [1, 4, 16]
        P_matrices = [butterfly_permutation_torch(64, s) for s in strides]
        for s, P in enumerate(P_matrices):
            self.register_buffer(f"P_{s}", P)

    def get_stage_matrix(self, stage: int) -> torch.Tensor:
        """Constructs the 64x64 orthogonal matrix for a single stage."""
        # Normalize quaternions to S^3 manifold
        l = self.qL[stage] / self.qL[stage].norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if self.use_b_rotors and self.qR is not None:
            r = self.qR[stage] / self.qR[stage].norm(dim=-1, keepdim=True).clamp_min(1e-12)
            R4 = cayley_matrix_torch(l, r)  # (16, 4, 4)
        else:
            R4 = left_matrix_torch(l)       # (16, 4, 4)

        # Build block-diagonal 64x64
        M = torch.zeros(64, 64, dtype=l.dtype, device=l.device)
        for b in range(16):
            M[4 * b:4 * b + 4, 4 * b:4 * b + 4] = R4[b]

        P = getattr(self, f"P_{stage}").to(dtype=l.dtype, device=l.device)
        # Stage transformation: B_s = P_s @ M
        return P @ M

    def get_full_matrix(self) -> torch.Tensor:
        """Computes the full 64x64 orthogonal matrix R = B_3 @ B_2 @ B_1."""
        R = torch.eye(64, dtype=self.qL.dtype, device=self.qL.device)
        for s in range(self.num_stages):
            B_s = self.get_stage_matrix(s)
            R = B_s @ R
        return R

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward transformation: x @ R^T
        Supports inputs of shape (..., 64).
        """
        R = self.get_full_matrix().to(dtype=x.dtype, device=x.device)
        return x @ R.T

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse adjoint transformation: y @ R
        Because R is strictly orthogonal, R^{-1} = R^T.
        """
        R = self.get_full_matrix().to(dtype=y.dtype, device=y.device)
        return y @ R


class QuATernaryonRotorNetwork(nn.Module):
    """
    Full Quaternion Rotor Network for an arbitrary dimension D that is a multiple of 64.
    Used for MiniCPM layers:
      - Attention (d = 2304 = 36 * 64) -> 36 parallel 64-dim butterfly blocks.
      - MLP intermediate (d = 5760 = 90 * 64) -> 90 parallel 64-dim butterfly blocks.
    """

    def __init__(self, dim: int, use_b_rotors: bool = True):
        super().__init__()
        assert dim % 64 == 0, f"Dimension {dim} must be a multiple of 64 for QuATernaryons."
        self.dim = dim
        self.num_heads_64 = dim // 64

        self.blocks = nn.ModuleList([
            QuaternionButterflyBlock64(use_b_rotors=use_b_rotors)
            for _ in range(self.num_heads_64)
        ])

    def get_full_matrix(self) -> torch.Tensor:
        """Constructs the full D x D block-diagonal orthogonal matrix."""
        R_blocks = [block.get_full_matrix() for block in self.blocks]
        return torch.block_diag(*R_blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward transform: x @ R^T.
        x: (..., D)
        """
        orig_shape = x.shape
        x_reshaped = x.reshape(-1, self.num_heads_64, 64)
        out_heads = []
        for h, block in enumerate(self.blocks):
            out_heads.append(block(x_reshaped[:, h, :]))
        out = torch.stack(out_heads, dim=1).reshape(*orig_shape)
        return out

    def inverse(self, y: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform: y @ R.
        y: (..., D)
        """
        orig_shape = y.shape
        y_reshaped = y.reshape(-1, self.num_heads_64, 64)
        out_heads = []
        for h, block in enumerate(self.blocks):
            out_heads.append(block.inverse(y_reshaped[:, h, :]))
        out = torch.stack(out_heads, dim=1).reshape(*orig_shape)
        return out
