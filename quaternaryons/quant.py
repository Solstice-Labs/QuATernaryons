"""
quaternaryons.quant — Asymmetric Ternary Quantizer with Low-Rank SVD Residual
Implements the sub-2-bit ternary core:
  W_q = beta (per-row offset) + repeat(alpha, g) * T + U V^T (rank-r correction)
Where T in {-1, 0, +1}, alpha in R^{out x (in/g)}, beta in R^{out x 1}, U in R^{out x r}, V in R^{in x r}.
"""

import math
from typing import Dict, Optional, Tuple
import numpy as np
import torch


def ternary_entropy(T: torch.Tensor) -> float:
    """Calculates empirical Shannon entropy of ternary codes in bits."""
    vals, counts = T.reshape(-1).unique(return_counts=True)
    p = counts.float() / counts.sum().float()
    ent = float(-(p * torch.log2(p + 1e-12)).sum().item())
    return ent


def quantize_ternary_group(
    W_slab: torch.Tensor,
    n_search: int = 24
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Grid-search optimal scale alpha for a slab of shape (N, g) minimizing ||W - alpha * T||_F^2.
    Returns:
        alpha: (N, 1) scale factor
        T: (N, g) ternary tensor in {-1, 0, +1}
    """
    amax = W_slab.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8)
    W_norm = W_slab / amax

    best_err = torch.full_like(amax, float("inf"))
    best_tau = torch.full_like(amax, 0.6)
    best_T = torch.zeros_like(W_slab)

    # Search tau in [0.35, 0.85]
    for tau_val in torch.linspace(0.35, 0.85, n_search, device=W_slab.device):
        T_cand = torch.clamp(torch.round(W_norm / tau_val), -1.0, 1.0)
        # LS scale refinement for this candidate: alpha = (W * T).sum() / (T * T).sum()
        num = (W_norm * T_cand).sum(dim=-1, keepdim=True)
        den = (T_cand * T_cand).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        alpha_cand = (num / den).clamp_min(1e-4)

        recon = alpha_cand * T_cand
        err = ((W_norm - recon) ** 2).sum(dim=-1, keepdim=True)

        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_tau = torch.where(better, tau_val, best_tau)
        best_T = torch.where(better.expand_as(best_T), T_cand, best_T)

    # Final refined alpha in unnormalized coordinates
    num = (W_slab * best_T).sum(dim=-1, keepdim=True)
    den = (best_T * best_T).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    alpha = (num / den).clamp_min(1e-6)

    return alpha, best_T.to(torch.int8)


def quantize_ternary_asymmetric(
    W: torch.Tensor,
    group_size: int = 32,
    n_search: int = 12
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fast fully-vectorized asymmetric ternary quantization:
        W ~ beta + repeat(alpha, g) * T
    Where beta is per-row mean, alpha is per-group scale, T in {-1, 0, +1}.
    Processes all groups simultaneously across entire matrix in a single vectorized pass.
    """
    out_dim, in_dim = W.shape
    assert in_dim % group_size == 0, f"in_dim {in_dim} must be divisible by group_size {group_size}"
    n_groups = in_dim // group_size

    beta = W.mean(dim=-1, keepdim=True)
    W_centered = (W - beta).view(-1, group_size)

    amax = W_centered.abs().max(dim=-1, keepdim=True).values.clamp_min(1e-8)
    W_norm = W_centered / amax

    best_err = torch.full_like(amax, float("inf"))
    best_T = torch.zeros_like(W_centered, dtype=torch.int8)

    for tau_val in torch.linspace(0.40, 0.80, n_search, device=W.device):
        T_cand = torch.clamp(torch.round(W_norm / tau_val), -1.0, 1.0)
        num = (W_norm * T_cand).sum(dim=-1, keepdim=True)
        den = (T_cand * T_cand).sum(dim=-1, keepdim=True).clamp_min(1e-6)
        alpha_cand = (num / den).clamp_min(1e-4)

        recon = alpha_cand * T_cand
        err = ((W_norm - recon) ** 2).sum(dim=-1, keepdim=True)

        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_T = torch.where(better, T_cand.to(torch.int8), best_T)

    num = (W_centered * best_T.float()).sum(dim=-1, keepdim=True)
    den = (best_T.float() * best_T.float()).sum(dim=-1, keepdim=True).clamp_min(1e-6)
    alpha = (num / den).clamp_min(1e-6)

    alpha = alpha.view(out_dim, n_groups)
    T = best_T.view(out_dim, in_dim)

    alpha_expanded = torch.repeat_interleave(alpha, group_size, dim=-1)
    W_base = beta + alpha_expanded * T.to(W.dtype)

    return beta, alpha, T, W_base


def fit_low_rank_residual(
    E: torch.Tensor,
    rank: int = 8,
    H: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fits rank-r correction factors U and V to minimize reconstruction error:
        min ||E - U V^T||_F^2  (or Hessian-weighted ||(E - U V^T) H^{1/2}||_F^2)
    Absorbs singular values equally: U <- U_r * sqrt(S_r), V <- V_r * sqrt(S_r).
    Returns:
        U: (out_dim, rank)
        V: (in_dim, rank)
    """
    if rank <= 0:
        device = E.device
        return torch.zeros(E.size(0), 0, device=device), torch.zeros(E.size(1), 0, device=device)

    if H is not None:
        # Whiten with Cholesky factor of Hessian: H = L L^T => E @ L
        eps = 1e-6 * torch.trace(H) / H.size(0)
        L = torch.linalg.cholesky(H + eps * torch.eye(H.size(0), dtype=H.dtype, device=H.device))
        target = E @ L
    else:
        target = E

    if rank <= 16 and target.size(0) >= 512:
        U, S, V = torch.svd_lowrank(target, q=rank, niter=4)
        r = min(rank, S.numel())
        sq = S[:r].sqrt()
        Ur = (U[:, :r] * sq.unsqueeze(0)).contiguous()
        Vr = (sq.unsqueeze(1) * V[:, :r].T).contiguous().T
    else:
        U, S, Vt = torch.linalg.svd(target, full_matrices=False)
        r = min(rank, S.numel())
        sq = S[:r].sqrt()
        Ur = (U[:, :r] * sq.unsqueeze(0)).contiguous()
        Vr = (sq.unsqueeze(1) * Vt[:r, :]).contiguous().T  # (in_dim, r)

    if H is not None:
        # Invert Cholesky transformation on Vr: Vr <- (Vr^T @ L^{-1})^T
        Vr = torch.linalg.solve_triangular(L.T, Vr.T, upper=True).T

    return Ur, Vr


def compute_bpw(out_dim: int, in_dim: int, group_size: int, rank: int, entropy: float) -> float:
    """Calculates exact honest bit-per-weight (bpw) including all metadata."""
    n_elements = out_dim * in_dim
    # Ternary code entropy + FP16 scales (16 bits/group) + FP16 offsets (16 bits/row) + FP16 SVD (16 * r * (out + in))
    bits_ternary = n_elements * entropy
    bits_alpha = (out_dim * (in_dim // group_size)) * 16.0
    bits_beta = out_dim * 16.0
    bits_svd = rank * (out_dim + in_dim) * 16.0
    total_bits = bits_ternary + bits_alpha + bits_beta + bits_svd
    return total_bits / n_elements
