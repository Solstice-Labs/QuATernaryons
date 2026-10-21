"""
tests/test_quaternaryons_math.py — Unit Tests for QuATernaryons Core Engine
Verifies algebraic properties, orthogonality, butterfly invertibility, and quantization fidelity.
"""

import math
import unittest
import torch
import torch.nn as nn

from quaternaryons import (
    qmul_torch,
    qconj_torch,
    left_matrix_torch,
    right_matrix_torch,
    cayley_matrix_torch,
    butterfly_permutation_torch,
    QuaternionButterflyBlock64,
    QuATernaryonRotorNetwork,
    quantize_ternary_asymmetric,
    fit_low_rank_residual,
    compute_bpw,
    QuATernaryonLinear
)


class TestQuATernaryons(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(20261021)

    def test_left_matrix_orthogonality_and_hamilton(self):
        # Sample random unit quaternion L
        l = torch.randn(4)
        l = l / l.norm()
        A = left_matrix_torch(l)

        # Check orthogonality: A^T @ A = I_4
        I4 = torch.eye(4)
        self.assertTrue(torch.allclose(A.T @ A, I4, atol=1e-6))
        self.assertTrue(torch.allclose(A @ A.T, I4, atol=1e-6))

        # Check Hamilton multiplication isomorphism: A(L) @ x == L (.) x
        x = torch.randn(4)
        y_mat = A @ x
        y_qmul = qmul_torch(l.unsqueeze(0), x.unsqueeze(0)).squeeze(0)
        self.assertTrue(torch.allclose(y_mat, y_qmul, atol=1e-6))

    def test_right_matrix_orthogonality(self):
        r = torch.randn(4)
        r = r / r.norm()
        B = right_matrix_torch(r)
        I4 = torch.eye(4)
        self.assertTrue(torch.allclose(B.T @ B, I4, atol=1e-6))
        self.assertTrue(torch.allclose(B @ B.T, I4, atol=1e-6))

    def test_cayley_factorization(self):
        l = torch.randn(4)
        l = l / l.norm()
        r = torch.randn(4)
        r = r / r.norm()
        R4 = cayley_matrix_torch(l, r)

        I4 = torch.eye(4)
        self.assertTrue(torch.allclose(R4.T @ R4, I4, atol=1e-6))
        det = torch.linalg.det(R4)
        self.assertTrue(torch.allclose(det, torch.tensor(1.0), atol=1e-5))

    def test_butterfly_64_orthogonality_and_invertibility(self):
        # 64 = 4^3
        block = QuaternionButterflyBlock64()
        # Randomize rotor parameters
        with torch.no_grad():
            block.qL.copy_(torch.randn_like(block.qL))
            block.qR.copy_(torch.randn_like(block.qR))

        R64 = block.get_full_matrix()
        I64 = torch.eye(64)
        # Check orthogonality of full 64x64 butterfly
        orth_err = (R64.T @ R64 - I64).abs().max().item()
        self.assertLess(orth_err, 1e-5, f"R64 not orthogonal, max err = {orth_err}")

        # Check forward and inverse match: inverse(forward(x)) == x
        x = torch.randn(5, 64)
        y = block(x)
        x_rec = block.inverse(y)
        inv_err = (x - x_rec).abs().max().item()
        self.assertLess(inv_err, 1e-5, f"Inverse roundtrip failed, max err = {inv_err}")

    def test_minicpm_dimensions_orthogonality(self):
        # MiniCPM Attention dim = 2304 (36 heads of 64)
        rotor_attn = QuATernaryonRotorNetwork(2304)
        x_attn = torch.randn(2, 8, 2304)
        y_attn = rotor_attn(x_attn)
        x_attn_rec = rotor_attn.inverse(y_attn)
        self.assertTrue(torch.allclose(x_attn, x_attn_rec, atol=1e-5))

        # MiniCPM MLP dim = 5760 (90 heads of 64)
        rotor_mlp = QuATernaryonRotorNetwork(5760)
        x_mlp = torch.randn(2, 4, 5760)
        y_mlp = rotor_mlp(x_mlp)
        x_mlp_rec = rotor_mlp.inverse(y_mlp)
        self.assertTrue(torch.allclose(x_mlp, x_mlp_rec, atol=1e-5))

    def test_asymmetric_ternary_quantization(self):
        # Weight shape 128 x 64
        W = torch.randn(128, 64) * 0.5
        # Add artificial mean offset to test asymmetric beta
        W += 1.5

        beta, alpha, T, W_base = quantize_ternary_asymmetric(W, group_size=32)
        # Verify ternary values
        unique_vals = T.unique()
        for v in unique_vals:
            self.assertIn(v.item(), [-1, 0, 1])

        # Verify low rank correction reduces error
        E = W - W_base
        base_err = (E ** 2).mean().item()
        U, V = fit_low_rank_residual(E, rank=8)
        E_corrected = E - U @ V.T
        svd_err = (E_corrected ** 2).mean().item()

        self.assertLess(svd_err, base_err, "Low-rank SVD did not reduce reconstruction error")

        # Verify bpw calculation on LLM-scale layer (e.g. MiniCPM 2304 x 2304)
        ent = 1.55
        bpw = compute_bpw(2304, 2304, group_size=32, rank=8, entropy=ent)
        self.assertGreater(bpw, 1.9)
        self.assertLess(bpw, 2.3)

    def test_quaternaryon_linear_replacement(self):
        # Pristine float linear layer (MiniCPM shape snippet: 64 x 64)
        lin = nn.Linear(64, 64, bias=True)
        qlin = QuATernaryonLinear.from_linear(lin, group_size=32, rank=4)

        x = torch.randn(4, 16, 64)
        y_q = qlin(x)
        self.assertEqual(y_q.shape, (4, 16, 64))
        self.assertFalse(torch.isnan(y_q).any())


if __name__ == "__main__":
    unittest.main()
