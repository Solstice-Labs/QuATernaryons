"""
quaternaryons.rotor — 4D Isoclinic Clifford Rotors and Cayley-Klein Geometry
Implements exact left-isoclinic A(L), right-isoclinic B(R), Cayley SO(4) factorizations,
and closed-form Rotor-Procrustes alignment.
"""

from typing import Tuple, Union
import numpy as np
import torch


def qmul_torch(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton quaternion product a (.) b; tensors of shape (..., 4)."""
    a0, a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    b0, b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return torch.stack([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ], dim=-1)


def qconj_torch(q: torch.Tensor) -> torch.Tensor:
    """Hamilton conjugate: q* = (q0, -q1, -q2, -q3)."""
    sgn = torch.tensor([1.0, -1.0, -1.0, -1.0], dtype=q.dtype, device=q.device)
    return q * sgn


def left_matrix_torch(l: torch.Tensor) -> torch.Tensor:
    """
    Constructs left-isoclinic matrix A(L) in SO(4) where A(L) @ x = L (.) x.
        A(L) = [[ l0, -l1, -l2, -l3],
                [ l1,  l0, -l3,  l2],
                [ l2,  l3,  l0, -l1],
                [ l3, -l2,  l1,  l0]]
    Shape: (..., 4) -> (..., 4, 4)
    """
    l0, l1, l2, l3 = l[..., 0], l[..., 1], l[..., 2], l[..., 3]
    row0 = torch.stack([l0, -l1, -l2, -l3], dim=-1)
    row1 = torch.stack([l1,  l0, -l3,  l2], dim=-1)
    row2 = torch.stack([l2,  l3,  l0, -l1], dim=-1)
    row3 = torch.stack([l3, -l2,  l1,  l0], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


def right_matrix_torch(r: torch.Tensor) -> torch.Tensor:
    """
    Constructs right-isoclinic matrix B(R) in SO(4) where B(R) @ x = x (.) R*.
        B(R) = [[ r0,  r1,  r2,  r3],
                [-r1,  r0, -r3,  r2],
                [-r2,  r3,  r0, -r1],
                [-r3, -r2,  r1,  r0]]
    Shape: (..., 4) -> (..., 4, 4)
    """
    r0, r1, r2, r3 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]
    row0 = torch.stack([ r0,  r1,  r2,  r3], dim=-1)
    row1 = torch.stack([-r1,  r0, -r3,  r2], dim=-1)
    row2 = torch.stack([-r2,  r3,  r0, -r1], dim=-1)
    row3 = torch.stack([-r3, -r2,  r1,  r0], dim=-1)
    return torch.stack([row0, row1, row2, row3], dim=-2)


def cayley_matrix_torch(l: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Full Cayley factorization of SO(4): R_4 = A(L) @ B(R)."""
    return left_matrix_torch(l) @ right_matrix_torch(r)


def rotor_procrustes_fit_torch(G: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Closed-form Rotor-Procrustes alignment (Lemma 3).
    Maximizes <A(L), G>_F subject to ||L||_2 = 1.
    S_k = <A(e_k), G>_F, L* = S / ||S||_2.
    """
    # e_k basis for k=0,1,2,3
    I4 = torch.eye(4, dtype=G.dtype, device=G.device)
    A_basis = left_matrix_torch(I4)  # (4, 4, 4)
    # Inner product S_k = (A_basis[k] * G).sum((-2, -1))
    S = (A_basis * G.unsqueeze(0)).sum(dim=(-2, -1))  # (4,)
    nrm = torch.linalg.norm(S)
    l_star = S / nrm if nrm > 1e-12 else torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=G.dtype, device=G.device)
    return l_star, nrm
