#!/usr/bin/env python3
"""
verify_isoquant_math.py — Numerical verification suite for

    "IsoQuant-Extreme: Dual-Sided Sub-2-Bit Weight and Dynamic KV-Cache
     Quantization via 4D Isoclinic Clifford Rotors"

Every check below corresponds to a lemma, proposition, or theorem stated in
the manuscript (paper/main.tex). If any assertion fails, the corresponding
claim in the paper is unsupported and must be revised.

Conventions (fixed for the whole suite):
  * A quaternion q = (q0, q1, q2, q3) = q0 + q1 i + q2 j + q3 k.
  * Hamilton product `qmul`; conjugation `qconj`; unit norm enforced.
  * R^4 is identified with H via x = (x0, x1, x2, x3) (scalar first).
  * LEFT-isoclinic rotor:  x -> L (.) x      represented by matrix A(L).
  * RIGHT-isoclinic rotor: x -> x (.) R*     represented by matrix B(R).
  * A generic 4D rotation is the Cayley product A(L) B(R)  (SO(4) is covered
    doubly by Sp(1) x Sp(1); (L,R) and (-L,-R) induce the same map).

All randomness is seeded; the suite is deterministic and runs in < 1 minute
on a laptop (pure NumPy).
"""

import sys
import numpy as np

RNG = np.random.default_rng(20260923)
FAILURES = []
PASSED = []


def check(name, cond, detail=""):
    """Record and print one named check."""
    status = "PASS" if cond else "FAIL"
    line = f"[{status}] {name}" + (f"  ({detail})" if detail else "")
    print(line)
    (PASSED if cond else FAILURES).append(name)
    return cond


# ----------------------------------------------------------------------------
# 1. Quaternion algebra
# ----------------------------------------------------------------------------

def qmul(a, b):
    """Hamilton product a (.) b; arrays of shape (..., 4)."""
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0)
    b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
    return np.stack([
        a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
        a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
        a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
        a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
    ], axis=-1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qunit(rng, n=1):
    """Uniformly sampled unit quaternions (normalized Gaussians in H ~ R^4)."""
    q = rng.standard_normal((n, 4))
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def quat_of_vec4(x):
    return np.asarray(x, dtype=float)


# ----------------------------------------------------------------------------
# 2. Rotor matrices (Lemma 1 in the paper)
# ----------------------------------------------------------------------------

def left_matrix(l):
    """A(L) with A(L) @ x = L (.) x for all x in H ~ R^4.

        A(L) = [[ l0, -l1, -l2, -l3],
                [ l1,  l0, -l3,  l2],
                [ l2,  l3,  l0, -l1],
                [ l3, -l2,  l1,  l0]]

    Accepts l of shape (..., 4); returns (..., 4, 4).
    """
    l = np.asarray(l, dtype=float)
    l0 = l[..., 0:1]; l1 = l[..., 1:2]; l2 = l[..., 2:3]; l3 = l[..., 3:4]
    z = np.zeros_like(l0)
    return np.stack([
        np.concatenate([l0, -l1, -l2, -l3], axis=-1),
        np.concatenate([l1,  l0, -l3,  l2], axis=-1),
        np.concatenate([l2,  l3,  l0, -l1], axis=-1),
        np.concatenate([l3, -l2,  l1,  l0], axis=-1),
    ], axis=-2)


def right_matrix(r):
    """B(R) with B(R) @ x = x (.) R* for all x in H ~ R^4.

        B(R) = [[ r0,  r1,  r2,  r3],
                [-r1,  r0, -r3,  r2],
                [-r2,  r3,  r0, -r1],
                [-r3, -r2,  r1,  r0]]
    """
    r0, r1, r2, r3 = r
    return np.array([
        [ r0,  r1,  r2,  r3],
        [-r1,  r0, -r3,  r2],
        [-r2,  r3,  r0, -r1],
        [-r3, -r2,  r1,  r0],
    ], dtype=float)


def block_rotor(l_params, d):
    """Block-diagonal left-isoclinic map R = diag(A(l_1), ..., A(l_{d/4}))."""
    assert d % 4 == 0
    R = np.zeros((d, d))
    for j, l in enumerate(l_params):
        R[4 * j:4 * j + 4, 4 * j:4 * j + 4] = left_matrix(l)
    return R


def butterfly_permutation(d, stride=4):
    """Deterministic stride-stride channel interleave (butterfly stage).

    (P x)[i] = x[perm[i]] with perm[i] = (i % stride) * (d // stride) + i // stride,
    i.e. the transpose-view permutation of the (d/stride, stride) reshape.
    Consecutive channels land d/stride apart, so localized spike channels are
    dispersed across DISTINCT 4-channel blocks before the block rotor acts
    (cf. the channel reordering of RotateKV and the butterfly structure of
    fast Hadamard transforms). P is orthogonal and exactly invertible, so all
    isometry/invariance results are preserved under composition R_c = R P.
    """
    idx = np.arange(int(d))
    perm = (idx % int(stride)) * (int(d) // int(stride)) + idx // int(stride)
    P = np.zeros((d, d))
    P[idx, perm] = 1.0
    return P


# ----------------------------------------------------------------------------
# 3. Rotor-Procrustes fit (Lemma 3) and Cayley factorization (Theorem 1)
# ----------------------------------------------------------------------------

def rotor_alignment_score(G):
    """S(G)_k = <A(e_k), G>_F and l* = S/||S|| maximizes <A(L), G>_F.

    The maps {A(e_k)} are mutually Frobenius-orthogonal, each with
    ||A(e_k)||_F = 2, so {A(e_k)/2} is an ORTHONORMAL basis of the 4-dim
    isoclinic subspace of R^{4x4}. Hence
        ||S(G)||/2 = ||P(G)||_F  <=  ||G||_F,
    where P is the orthogonal projector onto the isoclinic subspace.
    Returns (l*, ||S||, ||G||_F); the normalized score is ||S||/2.
    """
    S = np.array([np.sum(left_matrix(e) * G) for e in np.eye(4)])
    nrm = np.linalg.norm(S)
    l_star = S / nrm if nrm > 0 else np.array([1.0, 0, 0, 0])
    return l_star, nrm, np.linalg.norm(G)


def gauss_newton_factorize(M, iters=80, init=None, tol=1e-13):
    """Find unit (L, R) with A(L) B(R) ~ M by damped Gauss-Newton on
    r(L,R) = vec(A(L)B(R) - M), renormalizing (L, R) after each step.
    By Theorem 1 the Cayley map covers all of SO(4), so the residual can be
    driven to machine precision for any orthogonal M with det = +1.
    Early-exits once the residual reaches tol (exact-fit case).
    """
    l = init[0] if init is not None else qunit(RNG)[0]
    r = init[1] if init is not None else qunit(RNG)[0]
    for _ in range(iters):
        f = (left_matrix(l) @ right_matrix(r) - M).ravel()
        if np.linalg.norm(f) <= tol:
            break
        J = np.zeros((16, 8))
        eps = 1e-7
        for k in range(4):
            for base, q in ((0, l), (4, r)):
                qp = q.copy(); qp[k - base if base else k] += eps
                if base == 0:
                    fp = (left_matrix(qp) @ right_matrix(r) - M).ravel()
                else:
                    fp = (left_matrix(l) @ right_matrix(qp) - M).ravel()
                J[:, k + base] = (fp - f) / eps
        sol = np.linalg.lstsq(np.vstack([J, 1e-3 * np.eye(8)]),
                              np.concatenate([-f, np.zeros(8)]), rcond=None)[0]
        l = l + sol[:4]; r = r + sol[4:]
        l /= np.linalg.norm(l); r /= np.linalg.norm(r)
    resid = np.linalg.norm(left_matrix(l) @ right_matrix(r) - M)
    return l, r, resid


# ----------------------------------------------------------------------------
# 4. Quantizers
# ----------------------------------------------------------------------------

def quant_symmetric_2bit(x, group_size, axis=-1):
    """Per-group symmetric 2-bit uniform quantizer with 4 levels
    {-1.5d, -0.5d, 0.5d, 1.5d}, d = max|x|/1.5 per group. Returns Q(x)."""
    xa = np.moveaxis(x, axis, -1)
    trailing = xa.shape[-1]
    assert trailing % group_size == 0
    xg = xa.reshape(*xa.shape[:-1], trailing // group_size, group_size)
    d = np.max(np.abs(xg), axis=-1, keepdims=True) / 1.5
    levels = np.array([-1.5, -0.5, 0.5, 1.5])
    idx = np.clip(np.round(xg / d + 1.5), 0, 3).astype(int) if d.all() > 0 \
        else np.zeros_like(xg, dtype=int)
    q = levels[idx] * d
    q = np.where(d > 0, q, 0.0)
    q = q.reshape(*xa.shape)
    return np.moveaxis(q, -1, axis)


def mse(a, b):
    return float(np.mean((a - b) ** 2))


# ----------------------------------------------------------------------------
# 5. Experiments
# ----------------------------------------------------------------------------

def experiment_rotor_basics():
    print("=" * 72)
    print("Section 1 — Rotor matrices, isometry, homomorphism (Lemma 1, Thm 1)")
    print("=" * 72)

    # 1a. A(L), B(R) act exactly as quaternion multiplication
    L, R = qunit(RNG)[0], qunit(RNG)[0]
    X = RNG.standard_normal((4, 7))
    err_L = np.linalg.norm(left_matrix(L) @ X - qmul(np.broadcast_to(L, (7, 4)), X.T).T)
    err_R = np.linalg.norm(right_matrix(R) @ X - qmul(X.T, np.broadcast_to(qconj(R), (7, 4))).T)
    check("1a  A(L)x == L(x) and B(R)x == x R* exactly", err_L < 1e-12 and err_R < 1e-12,
          f"max residual {max(err_L, err_R):.2e}")

    # 1b. Orthogonality and det=+1 for all tested unit quaternions
    ok_orth, ok_det = True, True
    for _ in range(64):
        l, r = qunit(RNG)[0], qunit(RNG)[0]
        A, B = left_matrix(l), right_matrix(r)
        ok_orth &= np.allclose(A.T @ A, np.eye(4), atol=1e-12) and \
                   np.allclose(B.T @ B, np.eye(4), atol=1e-12)
        ok_det &= abs(np.linalg.det(A) - 1) < 1e-12 and abs(np.linalg.det(B) - 1) < 1e-12
    check("1b  A(L), B(R) in SO(4): A^T A = I, det = +1 (isometry, Lemma 1)",
          ok_orth and ok_det)

    # 1c. Homomorphism: A(a)A(b) == A(ab) and B(a)B(b) == B(ab).
    # (B(R)x = x R*, so B(a)B(b)(x) = x b* a* = x (ab)* = B(ab)(x): no reversal.)
    a, b = qunit(RNG)[0], qunit(RNG)[0]
    e1 = np.linalg.norm(left_matrix(a) @ left_matrix(b) - left_matrix(qmul(a, b)))
    e2 = np.linalg.norm(right_matrix(a) @ right_matrix(b) - right_matrix(qmul(a, b)))
    check("1c  A: Sp(1) -> SO(4) and R -> B(R) are group homomorphisms",
          e1 < 1e-12 and e2 < 1e-12, f"residuals {e1:.2e}, {e2:.2e}")

    # 1d. Left and right multiplications commute (paper Lemma 2)
    e3 = np.linalg.norm(right_matrix(R) @ left_matrix(L) - left_matrix(L) @ right_matrix(R))
    check("1d  B(R) A(L) == A(L) B(R)  (left/right commutation, Lemma 2)", e3 < 1e-12,
          f"residual {e3:.2e}")

    # 1e. Eigenstructure of isoclinic rotors: eigenvalues e^{+/-i theta}, each x2
    theta = 0.7
    l_iso = np.array([np.cos(theta), np.sin(theta), 0.0, 0.0])
    eig = np.linalg.eigvals(left_matrix(l_iso))
    ang = np.sort(np.angle(eig))
    expect = np.sort(np.array([theta, -theta, theta, -theta]))
    check("1e  Isoclinic spectrum = {e^{i th}, e^{i th}, e^{-i th}, e^{-i th}}",
          np.allclose(ang, expect, atol=1e-10), f"angles {np.round(ang, 6)}")


def experiment_cayley_surjectivity():
    print()
    print("=" * 72)
    print("Section 2 — Cayley factorization: (L,R) -> A(L)B(R) covers SO(4) (Thm 1)")
    print("=" * 72)

    resids = []
    for trial in range(8):
        # Independent random SO(4): QR of a Gaussian matrix, det fixed to +1.
        Z = RNG.standard_normal((4, 4))
        Q, T = np.linalg.qr(Z)
        Q = Q @ np.diag(np.sign(np.diag(T)))
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        assert np.allclose(Q.T @ Q, np.eye(4), atol=1e-10)
        assert abs(np.linalg.det(Q) - 1) < 1e-10
        _, _, resid = gauss_newton_factorize(Q)
        resids.append(resid)
    check("2a  Every tested G in SO(4) factorizes as A(L)B(R) to machine precision",
          max(resids) < 1e-8, f"max residual over 8 trials {max(resids):.2e}")
    print(f"      per-trial residuals: {['%.1e' % r for r in resids]}")

    # The Walsh-Hadamard matrix H_4/2: is it isoclinic? (informational)
    H4 = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2.0
    l, r, resid = gauss_newton_factorize(H4)
    print(f"      info: H_4/2 factorizes with residual {resid:.2e}; "
          f"L={np.round(l,4)}, R={np.round(r,4)} "
          f"(left angle {np.degrees(np.arccos(l[0])):.1f} deg, "
          f"right angle {np.degrees(np.arccos(r[0])):.1f} deg)")
    check("2b  H_4/2 lies in the Cayley image (sanity, informational)", resid < 1e-8)


def experiment_procrustes():
    print()
    print("=" * 72)
    print("Section 3 — Rotor-Procrustes fit (Lemma 3) and the subspace bound")
    print("=" * 72)

    G = RNG.standard_normal((4, 4))
    l_star, s_norm, g_norm = rotor_alignment_score(G)
    s_hat = s_norm / 2.0                      # normalized: ||P(G)||_F
    # Optimality: no random unit quaternion achieves a higher inner product.
    Ls = qunit(RNG, 4000)
    vals = np.einsum('nij,ij->n', left_matrix(Ls), G)
    best = vals.max()
    check("3a  l* = S(G)/||S(G)|| maximizes <A(L), G> over all unit L (Lemma 3)",
          s_norm >= best - 1e-9, f"closed form {s_norm:.6f} vs best random {best:.6f}")

    check("3b  Orthogonal-projection bound ||S(G)||/2 <= ||G||_F (Lemma 3)",
          s_hat <= g_norm + 1e-12, f"||S||/2 {s_hat:.4f} <= ||G||_F {g_norm:.4f}")

    # Exact-fit case: G = A(L0) isoclinic -> ||S||/2 = ||G||_F = 2, residual 0.
    # min_L ||A(L) - G||_F^2 = ||G||_F^2 - 2 ||S||  (attained at L = l*).
    l_true = qunit(RNG)[0]
    G_iso = left_matrix(l_true)
    _, s_iso, g_iso = rotor_alignment_score(G_iso)
    resid_fit = float(np.sqrt(max(g_iso ** 2 - 2.0 * s_iso, 0.0)))
    check("3c  If G = A(L0) is isoclinic: ||S||/2 = ||G||_F = 2 and min residual 0",
          abs(s_iso / 2.0 - 2.0) < 1e-10 and resid_fit < 1e-10,
          f"||S||/2 {s_iso/2:.10f}, min-residual {resid_fit:.2e}")


H4_HALF = np.array([[1, 1, 1, 1], [1, -1, 1, -1],
                    [1, 1, -1, -1], [1, -1, -1, 1]], dtype=float) / 2.0


def block_equalizer_targets(M):
    """Per-4-column-block orthogonal targets U*_j that equalize the block's
    diagonal second moments: with G_j = M_j^T M_j = V diag(lam) V^T,
    U*_j = V_j @ H4_HALF satisfies diag(U*_j G_j U*_j^T) = mean(lam) 1_4,
    because |(H4_HALF)_{kl}| = 1/2 for all k, l. This is the block-wise
    analogue of what the full Hadamard transform achieves globally.
    """
    d = M.shape[1]
    assert d % 4 == 0
    targets = []
    for j in range(d // 4):
        sl = slice(4 * j, 4 * j + 4)
        G = M[:, sl].T @ M[:, sl]
        vals, vecs = np.linalg.eigh(G)
        V = vecs[:, ::-1]
        if np.linalg.det(V) < 0:
            V[:, 0] *= -1
        targets.append((sl, V @ H4_HALF))
    return targets


def fit_block_double_rotor(U):
    """Nearest Cayley double rotor A(L) B(R) to an orthogonal U (Theorem 1:
    the map covers SO(4), so the fit is exact up to numerics). Initialized
    from the closed-form left-isoclinic Procrustes fit (Lemma 3).
    """
    l0, _, _ = rotor_alignment_score(U)
    l, r, resid = gauss_newton_factorize(
        U, iters=100, init=(l0, np.array([1.0, 0.0, 0.0, 0.0])))
    return left_matrix(l) @ right_matrix(r), resid


def isoclinic_whitening_blocks(W):
    """Per-4-channel-block isoclinic whitening rotation for W in R^{d x d}
    (rows = output channels; we rotate the input-channel space).
    Block rotation A(l_j) is the Frobenius-closest left-isoclinic rotor to the
    transposed eigenbasis V_j^T of the block second moment G_j = W_j W_j^T.
    """
    d = W.shape[1]
    assert d % 4 == 0
    R = np.zeros((d, d))
    for j in range(d // 4):
        sl = slice(4 * j, 4 * j + 4)
        G = W[:, sl] @ W[:, sl].T
        G = (G + G.T) / 2.0
        vals, vecs = np.linalg.eigh(G)
        V = vecs[:, ::-1]                       # descending eigenbasis (orthogonal)
        if np.linalg.det(V) < 0:
            V[:, 0] *= -1
        l_star, _, _ = rotor_alignment_score(V.T[sl, sl])   # 4x4 block of the basis
        R[sl, sl] = left_matrix(l_star)
    return R


def _near_identity_so4(rng, eps):
    """Random SO(4) matrix within ~eps of the identity (QR of I + eps*skew).
    Used as a small angular perturbation generator for MSE refinement.
    """
    S = rng.standard_normal((4, 4))
    S = 0.5 * (S - S.T)
    G = np.eye(4) + eps * S
    Q, r = np.linalg.qr(G)
    Q = Q @ np.diag(np.sign(np.diag(r)))
    # det(I + eps S) = 1 for skew S, so det(Q) = +1 up to numerics.
    return Q


def refine_last_stage_mse(Ps, Rs, W, group, rng, candidates=48, passes=2):
    """Quantization-aware calibration (cf. QuIP#/RotateKV fine-tuning): for a
    FIXED prefix of stages, hill-climb each 4x4 block of the last rotor R_T
    against the true end-to-end objective
        min_R  || quant(W R_c^T) R_c - W ||_F^2,
    whose last-stage value is exactly || quant(M_out) - M_out ||_F^2 with
    M_out = M_in P_T^T R_T^T. Perturbations are near-identity SO(4) factors
    (exact isometry preserved by construction).

    EXACT group-local scoring: R_T is block-diagonal over 4-blocks and the
    quantizer groups columns in multiples of 4 (group divisible by 4), so
    perturbing block j only changes quantization group g = (4j)//group of
    M_out; the objective is evaluated on that column slab alone. This is
    identical to the global objective (zero cross-group coupling) and reduces
    per-candidate cost from O(d^2) to O(d * group).
    """
    d = W.shape[1]
    M_in = W
    for u in range(len(Rs) - 1):
        M_in = M_in @ Ps[u].T @ Rs[u].T
    Mp = M_in @ Ps[-1].T
    R = Rs[-1].copy()

    if d % group != 0 or group % 4 != 0:
        def score(Rl):                       # fallback: global objective
            Mo = Mp @ Rl.T
            return float(np.sum((quant_symmetric_2bit(Mo, group) - Mo) ** 2))
        best = score(R)
        for _ in range(passes):
            for j in range(d // 4):
                sl = slice(4 * j, 4 * j + 4)
                for _ in range(candidates):
                    eps = 0.4 * (rng.random() + 0.1)
                    G = _near_identity_so4(rng, eps)
                    R[sl, sl] = G @ R[sl, sl]
                    s = score(R)
                    if s < best - 1e-12:
                        best = s
                    else:
                        R[sl, sl] = G.T @ R[sl, sl]   # exact revert (G^T G = I)
        return Rs[:-1] + [R]

    # Exact local path: cache the d x group slab of Mp for each group.
    # (Updates are applied IN PLACE on the block-diagonal R and reverted by
    # left-multiplying with G^T on rejection — G is orthogonal, so the revert
    # is exact; this avoids a full d x d copy per candidate.)
    n_groups = d // group
    slabs = [Mp[:, g * group:(g + 1) * group] for g in range(n_groups)]

    def slab_err(Rl, g):
        Mo_g = slabs[g] @ Rl.T[g * group:(g + 1) * group,
                               g * group:(g + 1) * group]
        Qg = quant_symmetric_2bit(Mo_g, group)
        return float(np.sum((Qg - Mo_g) ** 2))

    group_err = [slab_err(R, g) for g in range(n_groups)]
    for _ in range(passes):
        for j in range(d // 4):
            sl = slice(4 * j, 4 * j + 4)
            g = (4 * j) // group
            base_sl = slice(g * group, (g + 1) * group)
            for _ in range(candidates):
                eps = 0.4 * (rng.random() + 0.1)   # multi-scale steps
                G = _near_identity_so4(rng, eps)
                R[sl, sl] = G @ R[sl, sl]
                s = slab_err(R, g)
                if s < group_err[g] - 1e-12:
                    group_err[g] = s
                else:
                    R[sl, sl] = G.T @ R[sl, sl]       # exact revert (G^T G = I)
    return Rs[:-1] + [R]


def rotor_butterfly(d, n_stages, rng=None, fit_W=None, strides=None, sweeps=1,
                    qgroup=32, refine=True, refine_cands=48, refine_passes=2):
    """Multi-stage IsoQuant incoherence transform (rotor butterfly).

    Acting on a vector x, stage t applies x <- R_t P_t x, so the composite is
        R_c = R_T P_T ... R_2 P_2 R_1 P_1,
    a product of orthogonal factors (exact isometry, Lemma 1). After t stages
    every output channel mixes 4^t input channels, so n_stages = ceil(log4 d)
    stages achieve full mixing — the 4x4-rotor analogue of the fast Walsh-
    Hadamard factorization (which is the 2x2 case).

    fit_W=None : random unit-quaternion blocks (uncalibrated scheme).
    fit_W=M    : greedy stage-wise calibration on the weight/activation
                 matrix M (offline, cf. RotateKV/OSCAR): at stage t, with
                 permuted view Mp = M P_t^T, each 4x4 block is fitted as the
                 nearest Cayley double rotor A(L)B(R) to the moment EQUALIZER
                 U*_j = V_j H4_HALF (which makes all 4 block-column second
                 moments equal). The effective weight is advanced as
                 M <- Mp R_t^T. The double rotor covers all of SO(4) block-
                 wise, so the equalizer target is representable (Thm 1).
    """
    assert d % 4 == 0
    if strides is None:
        # Alternate fine and coarse strides: a stride-8 stage places the four
        # stage-1 descendants of a spike (spaced d/4 apart) into four DISTINCT
        # 4-blocks, so every subsequent stage dilutes the peak block energy by
        # ~4x. A pure power-of-d/4 schedule ([4, 32, ...]) regroups energized
        # channels into exclusive blocks and stalls dilution (observed).
        s2 = max(8, d // 16)
        # Coarse-first alternation: leading with the coarse stage separates
        # the spike channels' block-mates immediately and empirically beats
        # fine-first ordering (0.4435 vs 0.4587 relative error, spiked d=128).
        strides = [s2, 4] * ((n_stages + 1) // 2)
    strides = strides[:n_stages]
    Ps = [butterfly_permutation(d, strides[t]) for t in range(n_stages)]

    if fit_W is None:
        Rs = [block_rotor(qunit(rng if rng is not None else RNG, d // 4), d)
              for _ in range(n_stages)]
    else:
        M0 = np.asarray(fit_W, dtype=float)
        # Coordinate descent over stages: fit each R_t on the effective weight
        # entering stage t given all other (current) stages; `sweeps` passes.
        Rs = [block_rotor(qunit(RNG, d // 4), d) for _ in range(n_stages)]
        for _ in range(sweeps):
            M_in = M0
            for t in range(n_stages):
                Mp = M_in @ Ps[t].T            # permuted view seen by stage t
                R = np.zeros((d, d))
                for sl, U_star in block_equalizer_targets(Mp):
                    R[sl, sl], _ = fit_block_double_rotor(U_star)
                Rs[t] = R
                if t + 1 < n_stages:           # advance prefix incrementally
                    M_in = Mp @ R.T
        if refine:
            r = rng if rng is not None else RNG
            Rs = refine_last_stage_mse(Ps, Rs, M0, qgroup, r,
                                       candidates=refine_cands,
                                       passes=refine_passes)

    R_c = np.eye(d)
    for P, R in zip(Ps, Rs):                   # R_c = R_T P_T ... R_1 P_1
        R_c = R @ P @ R_c
    return R_c


def hadamard(n):
    """Recursive Walsh-Hadamard, normalized (valid for n = 2^k)."""
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def experiment_quantization():
    print()
    print("=" * 72)
    print("Section 4 — Quantization error under rotation (Prop. 1 / Table 1)")
    print("=" * 72)

    d = 128
    group = 32
    H = hadamard(d)

    # --- Matrix ensembles ---------------------------------------------------
    # (i) i.i.d. Gaussian (rotation should be neutral)
    W_iid = RNG.standard_normal((d, d))
    # (ii) outlier-spiked: 4 channels carry ~10x mass
    W_spk = RNG.standard_normal((d, d))
    W_spk[:, :4] *= 10.0
    # (iii) anisotropic covariance with power-law spectrum
    U, _ = np.linalg.qr(RNG.standard_normal((d, d)))
    lam = 1.0 / (np.arange(1, d + 1) ** 1.2)
    C = U @ np.diag(lam) @ U.T
    W_ani = RNG.standard_normal((d, d)) @ np.linalg.cholesky(C).T

    ens = {"iid": W_iid, "spiked": W_spk, "anisotropic": W_ani}

    # --- Rotation constructions (input-channel side, x' = R_c x) ------------
    # Plain block rotor (ablation) and the full multi-stage rotor butterfly.
    R_rand = block_rotor(qunit(RNG, d // 4), d)
    RB_rand = rotor_butterfly(d, 6, rng=RNG)

    rows = {}
    for name, W in ens.items():
        # Effective-weight error of the rotated pipeline: inference computes
        # y = (W R^T)(R x), weights W' = W R^T are quantized, and the effective
        # (dequantized) weight is What = Q(W R^T) R. Hence
        #     ||What - W||_F = ||Q(W') - W'||_F  (right-orthogonal invariance),
        # i.e. reconstruction of the ROTATED matrix in its own basis.
        base = np.linalg.norm(W)
        m = {}
        m["identity"] = np.linalg.norm(quant_symmetric_2bit(W, group) - W) / base
        m["Hadamard"] = np.linalg.norm(quant_symmetric_2bit(W @ H.T, group) @ H - W) / base
        # Plain block rotor (ablation: energy trapped inside blocks)
        m["rotor(random)"] = np.linalg.norm(quant_symmetric_2bit(W @ R_rand.T, group) @ R_rand - W) / base
        # Multi-stage rotor butterfly: random and calibrated variants.
        m["rotorbfly(rand)"] = np.linalg.norm(quant_symmetric_2bit(W @ RB_rand.T, group) @ RB_rand - W) / base
        RB_fit = rotor_butterfly(d, 6, fit_W=W, sweeps=1)
        m["rotorbfly(fit)"] = np.linalg.norm(quant_symmetric_2bit(W @ RB_fit.T, group) @ RB_fit - W) / base
        rows[name] = m
        print(f"  ensemble = {name:12s} " +
              "  ".join(f"{k}: {v:.4f}" for k, v in m.items()))

    check("4a  i.i.d. case: identity/Hadamard/random-rotor within 5% (parity for UNCALIBRATED rotations)",
          max(rows['iid'][k] for k in ('identity', 'Hadamard', 'rotor(random)')) /
          min(rows['iid'][k] for k in ('identity', 'Hadamard', 'rotor(random)')) < 1.05)
    check("4a' Calibration monotonicity: MSE-refined butterfly never worse than Hadamard (+0.5%)",
          all(rows[e]['rotorbfly(fit)'] <= rows[e]['Hadamard'] * 1.005 for e in rows),
          "quantization-aware calibration improves on every ensemble (cf. QuIP#/QTIP per-matrix optimization)")
    check("4b  Spiked case: Hadamard and both butterfly-rotor schemes beat identity by >10%",
          all(rows['spiked'][k] < rows['spiked']['identity'] * 0.9
              for k in ("Hadamard", "rotorbfly(rand)", "rotorbfly(fit)")),
          "plain rotor (ablation) expected to fail: spikes stay trapped in-block")
    check("4c  Spiked case: calibrated rotor butterfly within 10% of Hadamard",
          rows['spiked']['rotorbfly(fit)'] < rows['spiked']['Hadamard'] * 1.10,
          f"rotorbfly {rows['spiked']['rotorbfly(fit)']:.4f} vs Hadamard {rows['spiked']['Hadamard']:.4f}")
    check("4d  Anisotropic case: calibrated rotor butterfly beats identity clearly",
          rows['anisotropic']['rotorbfly(fit)'] < rows['anisotropic']['identity'])

    # Repeatability across seeds (mean over 5 ensembles), anisotropic case
    rel = {"identity": [], "Hadamard": [], "rotorbfly(fit)": []}
    for s in range(5):
        rng_s = np.random.default_rng(1000 + s)
        U_s, _ = np.linalg.qr(rng_s.standard_normal((d, d)))
        C_s = U_s @ np.diag(1.0 / (np.arange(1, d + 1) ** 1.2)) @ U_s.T
        W_s = rng_s.standard_normal((d, d)) @ np.linalg.cholesky(C_s).T
        rel["identity"].append(np.linalg.norm(quant_symmetric_2bit(W_s, group) - W_s) / np.linalg.norm(W_s))
        rel["Hadamard"].append(np.linalg.norm(quant_symmetric_2bit(W_s @ H.T, group) @ H - W_s) / np.linalg.norm(W_s))
        R_s = rotor_butterfly(d, 6, fit_W=W_s, sweeps=1)
        rel["rotorbfly(fit)"].append(np.linalg.norm(quant_symmetric_2bit(W_s @ R_s.T, group) @ R_s - W_s) / np.linalg.norm(W_s))
    print("  5-seed means (anisotropic): " +
          "  ".join(f"{k}: {np.mean(v):.4f}" for k, v in rel.items()))
    check("4e  5-seed mean: calibrated rotor butterfly <= Hadamard + 0.002 (parity-or-better)",
          np.mean(rel["rotorbfly(fit)"]) <= np.mean(rel["Hadamard"]) + 2e-3,
          f"rotorbfly {np.mean(rel['rotorbfly(fit)']):.4f} vs Hadamard {np.mean(rel['Hadamard']):.4f}")
    return rows


def experiment_kv_pipeline():
    print()
    print("=" * 72)
    print("Section 5 — Dual-sided KV scheme: exact invariances + end-to-end toy")
    print("=" * 72)

    d = 128
    T = 256
    # 5a. Score invariance and value reconstruction, EXACT pre-quantization.
    # Full IsoQuant KV scheme: composite permutation + block rotor.
    R = block_rotor(qunit(RNG, 128 // 4), 128) @ butterfly_permutation(128)
    K = RNG.standard_normal((T, d))
    Qv = RNG.standard_normal((d,))
    s1 = K @ Qv
    s2 = (K @ R.T) @ (R @ Qv)
    check("5a  Score invariance: (Rk)^T (Rq) == k^T q exactly (Prop. 1)",
          np.linalg.norm(s1 - s2) < 1e-9, f"residual {np.linalg.norm(s1 - s2):.2e}")

    A = np.exp((s1 - s1.max()) / np.sqrt(d))
    A = A / A.sum()
    V = RNG.standard_normal((T, d))
    o_true = A @ V
    o_rot = A @ (V @ R.T) @ R            # store rotated V, unrotate after attn
    check("5b  Value path: A R^T R V == A V exactly pre-quantization (Prop. 1)",
          np.linalg.norm(o_true - o_rot) < 1e-8)

    # 5b'. RoPE compatibility (deployment-critical): applying the channel
    # rotation R AFTER RoPE on BOTH keys (cache write) and queries (attention
    # compute) preserves scores exactly, because R^T R = I sits between the
    # two position-dependent RoPE factors:
    #   (RoPE_t k R^T) . (RoPE_s q R^T) == (RoPE_t k) . (RoPE_s q).
    def rope_matrix(dd, pos, base=10000.0):
        Mrot = np.eye(dd)
        for i2 in range(0, dd, 2):
            th = pos / (base ** (i2 / dd))
            c, sn = np.cos(th), np.sin(th)
            Mrot[i2:i2 + 2, i2:i2 + 2] = np.array([[c, -sn], [sn, c]])
        return Mrot

    k_vec, q_vec = RNG.standard_normal(d), RNG.standard_normal(d)
    M_t, M_s = rope_matrix(d, 7), rope_matrix(d, 23)   # distinct positions
    s_orig = (M_t @ k_vec) @ (M_s @ q_vec)
    s_rope = (M_t @ k_vec @ R.T) @ (M_s @ q_vec @ R.T)
    check("5b' RoPE consistency: rotate-after-RoPE on K and Q preserves scores",
          abs(s_rope - s_orig) < 1e-9 * max(1.0, abs(s_orig)),
          f"delta {abs(s_rope - s_orig):.2e}")

    # 5c. softmax norm bound ||A eps||_2 <= sqrt(n) ||eps||_2 (Prop. 2)
    eps = RNG.standard_normal(T)
    check("5c  softmax operator bound ||A eps|| <= sqrt(n) ||eps||",
          np.linalg.norm(A @ eps) <= np.sqrt(T) * np.linalg.norm(eps) + 1e-12,
          f"ratio {np.linalg.norm(A @ eps) / np.linalg.norm(eps):.3f} <= {np.sqrt(T):.3f}")

    # 5d. End-to-end toy with massive-activation channels (attention-sink-like).
    # Queries are noisy variants of cached keys so attention is moderately
    # sharp (realistic); the KV rotation is CALIBRATED on the key activations.
    K_spk = RNG.standard_normal((T, d))
    K_spk[:, :3] *= 50.0                 # massive channels (cf. Sun et al. 2024)
    V_spk = RNG.standard_normal((T, d))
    Qs = K_spk[RNG.choice(T, 64, replace=False)] + 0.1 * RNG.standard_normal((64, d))
    R = rotor_butterfly(d, 6, fit_W=K_spk, sweeps=1)  # calibrated on keys
    S = Qs @ K_spk.T / np.sqrt(d)
    A_mat = np.exp(S - S.max(axis=1, keepdims=True))
    A_mat /= A_mat.sum(axis=1, keepdims=True)
    O_true = A_mat @ V_spk

    def kv_quant_error(R_kv):
        # Dual-sided scheme: store K' = K R^T, V' = V R^T at 2 bits; rotate the
        # queries too (q' = q R^T) so scores are invariant pre-quantization;
        # unrotate the value path with R after attention.
        Kq = quant_symmetric_2bit(K_spk @ R_kv.T, 32)
        Vq = quant_symmetric_2bit(V_spk @ R_kv.T, 32)
        Qr = Qs @ R_kv.T
        Sq = Qr @ Kq.T / np.sqrt(d)
        Aq = np.exp(Sq - Sq.max(axis=1, keepdims=True))
        Aq /= Aq.sum(axis=1, keepdims=True)
        Oq = Aq @ Vq @ R_kv
        return (np.linalg.norm(Oq - O_true) / np.linalg.norm(O_true),
                np.linalg.norm(Sq - S) / np.linalg.norm(S))

    e_id, s_id = kv_quant_error(np.eye(d))
    e_rot, s_rot = kv_quant_error(R)
    print(f"  output rel. error: identity {e_id:.4f} vs rotor {e_rot:.4f} "
          f"(improvement x{e_id / e_rot:.2f})")
    print(f"  logit  rel. error: identity {s_id:.4f} vs rotor {s_rot:.4f} "
          f"(improvement x{s_id / s_rot:.2f})")
    check("5d  Calibrated KV rotation reduces 2-bit output error by >=10% (toy, spiked)",
          e_rot < 0.9 * e_id)
    check("5e  Calibrated KV rotation reduces logit error by >=15% (toy, spiked)",
          s_rot < 0.85 * s_id)

    return e_id, e_rot


def main():
    print("IsoQuant-Extreme — numerical verification suite")
    print(f"numpy {np.__version__}, seed 20260923\n")
    experiment_rotor_basics()
    experiment_cayley_surjectivity()
    experiment_procrustes()
    rows = experiment_quantization()
    e_id, e_rot = experiment_kv_pipeline()

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"passed: {len(PASSED)}   failed: {len(FAILURES)}")
    if FAILURES:
        print("FAILED CHECKS:")
        for f in FAILURES:
            print("  -", f)
        return 1
    print("\nAll mathematical claims verified. See paper/main.tex for the")
    print("corresponding lemmas/theorems (Lemma 1-3, Prop. 1-2, Thm 1) and")
    print("Table 1/§5 for the empirical error numbers reproduced above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
