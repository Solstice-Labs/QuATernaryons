#!/usr/bin/env python3
"""Modal runner: ternary+rank rel2 on REAL SmolLM2 weights + REAL Hessians.
12 modules x {r0, r8/16/32/64 ternary, r8 fp16} on one T4.
Review fixes included: rotated-space Hessian metric, sqrt(s) absorption,
fp32 tern_ls. Results -> /results/s1_*.json on volume isoq-data."""
import json, os, time
import modal

app = modal.App("isoq-s1")
volume = modal.Volume.from_name("isoq-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.4.1", "safetensors", "huggingface_hub", "numpy")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0"})
)

@app.function(gpu="T4", image=image, timeout=3600, volumes={"/data": volume},
              memory=16384)
def run_module(role: str, li: int) -> dict:
    import torch, glob, time
    from safetensors.torch import load_file
    torch.manual_seed(20260924)
    t0 = time.time()
    P = lambda m: print(f"  [{m}] {time.time()-t0:.1f}s", flush=True)

    snap = glob.glob("/root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B/snapshots/*")
    if not snap:
        from huggingface_hub import snapshot_download
        snap = [snapshot_download("HuggingFaceTB/SmolLM2-1.7B")]
    st = glob.glob(snap[0] + "/*.safetensors")[0]
    key = f"model.layers.{li}.self_attn.{role}.weight"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{role} {li} device={dev}", flush=True)
    W = load_file(st)[key].float().to(dev); P("W")
    H = torch.load(f"/data/hessians/model_layers_{li}_self_attn_{role}.pt",
                   map_location=dev, weights_only=False)
    if isinstance(H, dict): H = next(iter(H.values()))
    H = H.float().to(dev); P("H")

    def hadamard(n):
        Hh = torch.ones(1, 1, dtype=torch.float32, device=dev)
        while Hh.shape[0] < n:
            Hh = torch.cat([torch.cat([Hh, Hh], 1), torch.cat([Hh, -Hh], 1)], 0)
        return Hh / (n ** 0.5)
    Hn = hadamard(2048)
    H_rot = Hn @ H @ Hn; P("H_rot")

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
        sh = X.shape
        Xf = X.reshape(-1, g)
        n = sh[0] * sh[1]
        if Xf.shape[0] * g != n:
            pad = (-n) % g
            Xf = torch.cat([X.reshape(-1),
                            torch.zeros(pad, dtype=X.dtype, device=X.device)])
            Xf = Xf[: ((n + g - 1) // g) * g].reshape(-1, g)
        amax = Xf.abs().max(1, keepdim=True).values.clamp_min(1e-12)
        Xn = Xf / amax
        Tn = torch.clamp(torch.round(Xn / 0.6), -1, 1)
        for _ in range(iters):
            num = (Xn * Tn).sum(1, keepdim=True)
            den = (Tn * Tn).sum(1, keepdim=True).clamp_min(1e-12)
            a = (num / den).clamp(1e-6, 1.0)
            Tn = torch.clamp(torch.round(Xn / a), -1, 1)
        return (a * Tn * amax).reshape(-1)[:n].reshape(sh)

    def eval_rel2(E, Hb):
        EH = E @ Hb
        v = float((EH.double() * E.double()).sum())
        del EH
        return v / float(torch.trace(Hb.double()))

    Q0, _ = quant_ternary(W)
    rA = eval_rel2(W - Q0, H); P("r0_norot")
    Wr = W @ Hn
    Qr, Hc = quant_ternary(Wr)
    E_rot = Wr - Qr
    rB = eval_rel2(E_rot, H_rot); P("r0_rot")
    M = torch.linalg.cholesky(H_rot + 1e-6 * torch.eye(2048, device=dev))
    C = E_rot @ M; del M; P("C")
    U, s, Vt = torch.linalg.svd(C, full_matrices=False); del C; P("svd")
    row = {"r0_norot": rA, "r0": rB, "Hc": Hc}
    for r in (8, 16, 32, 64):
        sq = s[:r].sqrt()
        Ur = (U[:, :r] * sq).contiguous()
        Vr = (sq.unsqueeze(1) * Vt[:r, :]).contiguous()
        Ut = tern_ls(Ur); Vt2 = tern_ls(Vr)
        E2 = E_rot - Ut @ Vt2
        row[f"r{r}_tern"] = eval_rel2(E2, H_rot); P(f"r{r}_tern")
        del Ur, Vr, Ut, Vt2, E2
    sq = s[:8].sqrt()
    Ur = (U[:, :8] * sq).contiguous()
    Vr = (sq.unsqueeze(1) * Vt[:8, :]).contiguous()
    row["r8_fp16"] = eval_rel2(E_rot - Ur @ Vr, H_rot); P("r8_fp16")
    print("ROW", json.dumps(row), flush=True)
    with open(f"/results/s1_{role}_{li}.json", "w") as f:
        json.dump(row, f)
    return row

@app.local_entrypoint()
def main():
    mods = [("q_proj", 0), ("v_proj", 0), ("k_proj", 0), ("o_proj", 0),
            ("q_proj", 12), ("v_proj", 12), ("q_proj", 23), ("v_proj", 23),
            ("k_proj", 23), ("o_proj", 23)]
    import json as J
    rows = {}
    for r in run_module.starmap(mods):     # sequential; T4 has plenty of headroom
        pass
    # starmap returns generator; collect properly
    for (role, li), row in zip(mods, run_module.starmap(mods)):
        rows[f"L{li}.{role}"] = row
    import numpy as np
    agg = {}
    for k in ["r0_norot", "r0", "r8_tern", "r16_tern", "r32_tern", "r64_tern", "r8_fp16"]:
        vals = [v[k] for v in rows.values()]
        agg[k] = dict(mean=float(np.mean(vals)), std=float(np.std(vals)),
                      min=float(np.min(vals)), max=float(np.max(vals)))
    print("\n==== S1 AGGREGATE (real weights, real Hessians) ====")
    for k, v in agg.items():
        print(f"  {k:9s} rel2 = {v['mean']:.3f} ± {v['std']:.3f}  [{v['min']:.3f},{v['max']:.3f}]")
    Hc = float(np.mean([v["Hc"] for v in rows.values()]))
    print(f"  mean ternary code entropy = {Hc:.3f} bits")
    out = dict(per_module=rows, agg=agg, Hc=Hc)
    with open("research/s1_modal_results.json", "w") as f:
        J.dump(out, f, indent=1, default=float)
    print("saved -> research/s1_modal_results.json")
