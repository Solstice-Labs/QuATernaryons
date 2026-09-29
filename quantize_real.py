#!/usr/bin/env python3
"""
quantize_real.py — IsoQuant-Extreme on REAL trained weights.

Streams tensors from a GGUF (mmap; no full-file load), extracts attention
projection matrices, and runs the exact pipeline verified in
verify_isoquant_math.py (seeded, deterministic):

  effective-weight error = || quant2bit(W R_c^T) R_c - W ||_F / ||W||_F

for R_c in {identity, Hadamard, random rotor butterfly, calibrated rotor
butterfly}. For attn_k / attn_v the rotation acts on the OUTPUT-channel side
(rows, kv-head layout), which is precisely the per-head transform fused into
KV-cache stores; for attn_o it acts on the INPUT side. This measures the
weight-side effect of the transform on real trained statistics; KV activation
calibration (needing runtime traces) is out of scope here and stated as such.

Run:  python3 quantize_real.py <path.gguf> [--layers 4] [--refine-cands 16]
Output: research/real_model_results.md (+ JSON alongside).
"""

import argparse
import json
import os
import re
import sys
import time

import numpy as np

from verify_isoquant_math import (
    hadamard,
    quant_symmetric_2bit,
    rotor_butterfly,
    RNG,
)

GROUP = 32
BITS = 2


def dequant_tensor(raw, ttype):
    """GGML tensor -> float32 numpy 2-D (gguf.dequantize with BF16 fallback)."""
    import gguf
    try:
        out = gguf.dequantize(np.asarray(raw), gguf.GGMLQuantizationType(ttype))
    except Exception:
        # Manual BF16 -> FP32 (truncate is round-to-nearest-safe here: shift up)
        if int(ttype) == int(gguf.GGMLQuantizationType.BF16):
            u = np.asarray(raw).view(np.uint16).astype(np.uint32) << 16
            out = u.view(np.float32)
        else:
            raise
    return np.asarray(out, dtype=np.float32)


def block_rotor_identity(d):
    """Orthogonal identity (explicit) — kept for clarity in composites."""
    return np.eye(d)


def per_head_rotor(R_full, d_out, n_heads):
    """Tile a per-head d_h x d_h rotor across the kv-head layout.
    Also verifies the tiled matrix is exactly orthogonal (isometry claim
    must survive tiling before we trust any number it produces)."""
    d_h = d_out // n_heads
    assert R_full.shape == (d_h, d_h) and n_heads * d_h == d_out
    Rt = np.zeros((d_out, d_out))
    for h in range(n_heads):
        Rt[h * d_h:(h + 1) * d_h, h * d_h:(h + 1) * d_h] = R_full
    assert np.allclose(Rt.T @ Rt, np.eye(d_out), atol=1e-10)
    return Rt


def pick_matrices(reader, n_layers, kinds):
    """Choose requested attn_* tensors from evenly spaced layers."""
    blk = {}
    for t in reader.tensors:
        m = re.match(r"blk\.(\d+)\.(attn_k|attn_v|attn_output)\.weight$", t.name)
        if m and m.group(2) in kinds:
            blk.setdefault(int(m.group(1)), {})[m.group(2)] = t
    if not blk:
        raise SystemExit("No blk.N.attn_*.weight tensors found — is this a llama.cpp GGUF?")
    layers = sorted(blk)
    idx = np.linspace(0, len(layers) - 1, n_layers).round().astype(int)
    chosen = [layers[i] for i in dict.fromkeys(idx)]
    out = []
    for n in chosen:
        for kind in ("attn_k", "attn_v", "attn_output"):
            if kind in blk.get(n, {}):
                out.append((n, kind, blk[n][kind]))
    return out


def rotate_side(shape, kind):
    """Return (rotated-dim, side) — output side for k/v (KV-store fuse,
    per-head block-diagonal), input side for attn_output (absorbed into the
    attention epilogue / next-layer input projection)."""
    d_out, d_in = shape
    if kind in ("attn_k", "attn_v"):
        return d_out, "rows"
    return d_in, "cols"


def effective_error(W, R, side, group=GROUP):
    """Relative error of the effective weight after rotation + 2-bit quant.
    The quantizer groups along the ROTATED channel axis: columns for the
    cols-side (W' = W R^T), rows for the rows/KV side (W' = R W), matching
    how stored KV channels are grouped at inference."""
    if side == "rows":                     # W' = R W ; eff = R^T Q_rows(W')
        Wq = quant_symmetric_2bit(R @ W, group, axis=0)
        return float(np.linalg.norm(R.T @ Wq - W) / np.linalg.norm(W))
    Wt = W @ R.T                           # W' = W R^T ; eff = Q(W') R
    Wq = quant_symmetric_2bit(Wt, group)
    return float(np.linalg.norm(Wq @ R - W) / np.linalg.norm(W))


def block_hadamard(d):
    """Orthogonal H for any d: blockdiag of pow2 Hadamards (e.g. 1536)."""
    parts, rem = [], d
    p = 1
    while p * 2 <= rem:
        p *= 2
    while rem > 0:
        parts.append(hadamard(p))
        rem -= p
        p = 1
        while p * 2 <= rem:
            p *= 2
    R = np.zeros((d, d))
    o = 0
    for H in parts:
        R[o:o + H.shape[0], o:o + H.shape[0]] = H
        o += H.shape[0]
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--refine-cands", type=int, default=16)
    ap.add_argument("--fit-rows", type=int, default=0,
                    help="subsample this many rows of the fit matrix for "
                         "calibration (0 = all); error evaluation always "
                         "uses the full matrix")
    ap.add_argument("--kinds", default="attn_k,attn_v,attn_output",
                    help="comma-separated tensor kinds to process")
    ap.add_argument("--out", default="research/real_model_results")
    args = ap.parse_args()
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]

    import gguf
    print(f"reading {args.gguf}", flush=True)
    reader = gguf.GGUFReader(args.gguf)
    tensors = [t for t in pick_matrices(reader, args.layers, set(kinds)) if t[1] in kinds]
    print(f"selected {len(tensors)} matrices across layers "
          f"{sorted({n for n, _, _ in tensors})}", flush=True)

    results = []
    for (n, kind, t) in tensors:
        shape = tuple(t.shape[::-1])       # gguf stores reversed; -> (d_out, d_in)
        W = dequant_tensor(t.data, t.tensor_type).reshape(shape)
        d, side = rotate_side(shape, kind)
        t0 = time.time()

        if side == "rows":
            n_heads = d // 128
            W_eff = W.T                    # [fan-in, rotated]: columns = rotated dim
        else:
            n_heads = None
            W_eff = W

        # Optional row subsampling for CALIBRATION only (fixed seed): second-
        # moment estimates from a few thousand rows are statistically ample,
        # and error evaluation below remains exact on the full matrix.
        if args.fit_rows and args.fit_rows > 0 and W_eff.shape[0] > args.fit_rows:
            ridx = np.sort(np.random.default_rng(7).choice(
                W_eff.shape[0], args.fit_rows, replace=False))
            W_fit = W_eff[ridx]
        else:
            W_fit = W_eff

        if n_heads is not None and d % 128 == 0:
            # Per-head 128x128 calibrated rotor butterfly (deployment-faithful,
            # matches KV-store fusion; 4 kv-heads x 128 in this architecture).
            R_head_rand = rotor_butterfly(128, 6, rng=RNG,
                                          strides=[max(8, 128 // 16), 4] * 3)
            R_rand = per_head_rotor(R_head_rand, d, n_heads)
            H = np.zeros((d, d))
            Hh = block_hadamard(128)
            for h in range(n_heads):
                H[h * 128:(h + 1) * 128, h * 128:(h + 1) * 128] = Hh
            heads = [slice(h * 128, (h + 1) * 128) for h in range(n_heads)]
            R_cal = np.zeros((d, d))
            for hi, sl in enumerate(heads):
                sub = W_fit[:, sl]           # (n_fit, 128) head slice: rotated dims on columns
                print(f"  calibrating blk{n}.{kind} head {hi+1}/{n_heads}", flush=True)
                R_cal[sl, sl] = rotor_butterfly(
                    128, 6, fit_W=sub, strides=[max(8, 128 // 16), 4] * 3,
                    qgroup=GROUP, refine=True,
                    refine_cands=args.refine_cands, refine_passes=1)
        else:
            assert d % 4 == 0 and d % GROUP == 0, \
                f"{kind}: rotated dim {d} must be divisible by 4 and {GROUP}"
            strides = [max(8, d // 16), 4] * 3
            R_rand = rotor_butterfly(d, 6, rng=RNG, strides=strides)
            print(f"  calibrating blk{n}.{kind} (d={d}, fit rows={W_fit.shape[0]})",
                  flush=True)
            R_cal = rotor_butterfly(d, 6, fit_W=W_fit, strides=strides,
                                    qgroup=GROUP, refine=True,
                                    refine_cands=args.refine_cands,
                                    refine_passes=1)
            H = block_hadamard(d)

        R_use = R_cal if side == "rows" else R_cal.T
        row = {
            "layer": int(n), "kind": kind, "side": side, "d_rot": int(d),
            "per_head": n_heads is not None,
            "shape": [int(x) for x in shape],
            "identity": effective_error(W, np.eye(d), side),
            "hadamard": effective_error(W, H, side),
            "rotor_rand": effective_error(W, R_rand, side),
            "rotor_calibrated": effective_error(W, R_use, side),
            "secs": round(time.time() - t0, 1),
        }
        results.append(row)
        print(f"blk{n}.{kind:10s} d={d:5d} {side:4s} "
              f"id {row['identity']:.4f}  H {row['hadamard']:.4f}  "
              f"rand {row['rotor_rand']:.4f}  cal {row['rotor_calibrated']:.4f} "
              f"({row['secs']}s)", flush=True)
        del W, W_eff, R_rand, R_use, R_cal, H   # keep RSS flat between tensors

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(results, f, indent=2)

    keys = ["identity", "hadamard", "rotor_rand", "rotor_calibrated"]
    mean = {k: float(np.mean([r[k] for r in results])) for k in keys}
    kv_rows = [r for r in results if r["kind"] in ("attn_k", "attn_v")]
    mean_kv = {k: float(np.mean([r[k] for r in kv_rows])) for k in keys}

    lines = [
        "# Real-model results: IsoQuant-Extreme pipeline on Bonsai-27B weights",
        "",
        f"Source: `{os.path.basename(args.gguf)}` (llama.cpp GGUF, mmap-streamed).",
        f"Pipeline: identical to verify_isoquant_math.py — 6-stage rotor butterfly,",
        f"coarse-first strides, moment-equalizer fit + quantization-aware refinement;",
        f"2-bit group-{GROUP} symmetric quantizer. k/v rotated on the OUTPUT side",
        f"(per-head KV-store fuse), o on the INPUT side.",
        "",
        "| layer | tensor | rotated dim | identity | Hadamard | rotor(rand) | rotor(calibrated) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['layer']} | {r['kind']} | {r['d_rot']} | {r['identity']:.4f} | "
            f"{r['hadamard']:.4f} | {r['rotor_rand']:.4f} | {r['rotor_calibrated']:.4f} |")
    lines += [
        "",
        f"**Means (all):** " + "  ".join(f"{k} {v:.4f}" for k, v in mean.items()),
        f"**Means (k/v only):** " + "  ".join(f"{k} {v:.4f}" for k, v in mean_kv.items()),
        "",
        "Scope note: weight-side reconstruction on real trained matrices. These are",
        "NOT model perplexities; KV-activation calibration requires runtime traces.",
    ]
    with open(args.out + ".md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines[-5:]))
    print(f"\nwrote {args.out}.md / .json")


if __name__ == "__main__":
    main()
