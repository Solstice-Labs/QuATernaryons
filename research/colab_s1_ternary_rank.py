#!/usr/bin/env python3
"""Colab S1 v2 — review fixes applied.
  (1) r0_rot: error kept in ROTATED space, evaluated vs H_rot = Hn H Hn^T.
      (Hn symmetric => Hn^T = Hn, but written explicitly for correctness.)
  (2) rank-r subspace fit IN ROTATED SPACE (no Vt@Hn), evaluated vs H_rot.
  (3) singular values absorbed: U_r <- U_r*sqrt(s), V_r <- sqrt(s)*Vt_r.
  (4) tern_ls in fp32.
One module per invocation:  python3 colab_s1_ternary_rank.py q_proj 0
Writes /content/s1_<role>_<li>.json
Metric: rel2 = tr(E H E^T)/tr(H) with the Hessian in the SAME basis as E.
"""
import sys, json, time
import torch

role, li = sys.argv[1], int(sys.argv[2])
torch.manual_seed(20260924)
t0 = time.time()
P = lambda m: print(f"  [{m}] {time.time()-t0:.1f}s", flush=True)

snap = __import__("glob").glob(
    "/root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B/snapshots/*")[0]
from safetensors.torch import load_file
st = __import__("glob").glob(snap + "/*.safetensors")[0]
key = f"model.layers.{li}.self_attn.{role}.weight"
W = load_file(st)[key].float().cpu(); P("W")
H = torch.load(f"/content/hessians/model_layers_{li}_self_attn_{role}.pt",
               map_location="cpu", weights_only=False)
if isinstance(H, dict): H = next(iter(H.values()))
H = H.float(); P("H")

def hadamard(n):
    Hh = torch.ones(1, 1, dtype=torch.float32)
    while Hh.shape[0] < n:
        Hh = torch.cat([torch.cat([Hh, Hh], 1), torch.cat([Hh, -Hh], 1)], 0)
    return Hh / (n ** 0.5)

Hn = hadamard(2048)          # symmetric + orthogonal: Hn = Hn^T = Hn^{-1}
H_rot = Hn @ H @ Hn          # Hn H Hn^T  (== Hn H Hn here; kept explicit)

def quant_ternary(W, g=32):
    out, inn = W.shape
    Wf = W.reshape(-1, g)
    amax = Wf.abs().max(1, keepdim=True).values.clamp_min(1e-12)
    Wn = Wf / amax
    def err(a):
        T = torch.clamp(torch.round(Wn / a), -1, 1)
        return ((Wn - a * T) ** 2).sum(1)
    lo = torch.full_like(amax, 1e-4); hi = torch.ones_like(amax)
    gr = (5 ** 0.5 - 1) / 2
    c = hi - gr * (hi - lo); d = lo + gr * (hi - lo)
    for _ in range(36):
        m = err(c) < err(d)
        hi = torch.where(m, d, hi); lo = torch.where(m, lo, c)
        c = hi - gr * (hi - lo); d = lo + gr * (hi - lo)
    a = 0.5 * (lo + hi)
    T = torch.clamp(torch.round(Wn / a), -1, 1)
    Q = (a * T * amax).reshape(out, inn)
    vals, cnts = T.reshape(-1).unique(return_counts=True)
    p = cnts.float() / cnts.sum()
    return Q, float(-(p * p.log2()).sum())

def tern_ls(X, g=32, iters=3):
    """ternary LS fit in fp32; X stays in fp32."""
    sh = X.shape
    Xf = X.reshape(-1, g)
    n = sh[0] * sh[1]
    if Xf.shape[0] * g != n:
        pad = (-n) % g
        Xf = torch.cat([X.reshape(-1).double(),
                        torch.zeros(pad, dtype=X.dtype)]).reshape(-1, g)
    amax = Xf.abs().max(1, keepdim=True).values.clamp_min(1e-12)
    Xn = Xf / amax
    Tn = torch.clamp(torch.round(Xn / 0.6), -1, 1)
    for _ in range(iters):
        num = (Xn * Tn).sum(1, keepdim=True)
        den = (Tn * Tn).sum(1, keepdim=True).clamp_min(1e-12)
        a = (num / den).clamp(1e-6, 1.0)
        Tn = torch.clamp(torch.round(Xn / a), -1, 1)
    Qf = a * Tn * amax
    return Qf.reshape(-1)[:n].reshape(sh)

def eval_rel2(E, Hb):
    """tr(E Hb E^T)/tr(Hb) — Hb must live in E's basis."""
    EH = E @ Hb
    v = float((EH.double() * E.double()).sum())
    del EH
    return v / float(torch.trace(Hb.double()))

# A: no rotation, orig basis
Q0, _ = quant_ternary(W)
rA = eval_rel2(W - Q0, H); P("r0_norot")

# B: rotated basis, error in rotated space, Hessian rotated to match  [FIX 1]
Wr = W @ Hn                                   # W R^T, R = Hn (symmetric)
Qr, Hc = quant_ternary(Wr)
E_rot = Wr - Qr                               # stays in ROTATED space
rB = eval_rel2(E_rot, H_rot); P("r0_rot")

# rank-r: subspace of E_rot H_rot^{1/2} IN ROTATED SPACE              [FIX 2]
M = torch.linalg.cholesky(H_rot + 1e-6 * torch.eye(2048))
C = E_rot @ M; del M; P("C")
U, s, Vt = torch.linalg.svd(C, full_matrices=False); del C; P("svd")

row = {"r0_norot": rA, "r0": rB, "Hc": Hc}
for r in (8, 16, 32, 64):
    sq = s[:r].sqrt()                                              # [FIX 3]
    Ur = (U[:, :r] * sq).contiguous()          # absorb sqrt(s) into U
    Vr = (sq.unsqueeze(1) * Vt[:r, :]).contiguous()  # and into V
    Ut = tern_ls(Ur); Vt2 = tern_ls(Vr)
    E2 = E_rot - Ut @ Vt2                        # still rotated space
    row[f"r{r}_tern"] = eval_rel2(E2, H_rot); P(f"r{r}_tern")
    del Ur, Vr, Ut, Vt2, E2
# fp16-factor cross-check, same spaces
sq = s[:8].sqrt()
Ur = (U[:, :8] * sq).contiguous()
Vr = (sq.unsqueeze(1) * Vt[:8, :]).contiguous()
row["r8_fp16"] = eval_rel2(E_rot - Ur @ Vr, H_rot); P("r8_fp16")

json.dump(row, open(f"/content/s1_{role}_{li}.json", "w"), default=float)
print(f"DONE {role} {li} -> /content/s1_{role}_{li}.json", flush=True)
