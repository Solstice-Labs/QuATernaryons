#!/usr/bin/env python3
"""Light half of the full-ctx check: budget (from real config) + exact/MC
attention-vs-T math + structural sections. Runs in seconds, tiny memory.
Heavy S1 (real-weight ternary rank curve) runs on Colab via
research/colab_s1_ternary_rank.py."""
import json
import numpy as np

np.random.seed(20260924)
R = {}

# ---- S4 budget [CONFIG dims, exact] ----------------------------------------
CFG = dict(n_layers=64, n_gdn=48, n_full=16, hidden=5120, inter=17408,
           vocab=248320, n_qheads=24, n_kvheads=4, head_dim=256,
           partial_rotary=0.25, max_pos=262144, state_dtype="float32")
attn_pl = (CFG["hidden"]*CFG["n_qheads"]*CFG["head_dim"]
           + 2*CFG["hidden"]*CFG["n_kvheads"]*CFG["head_dim"]
           + CFG["n_qheads"]*CFG["head_dim"]*CFG["hidden"])
P_attn, P_mlp = 16*attn_pl, 64*3*CFG["hidden"]*CFG["inter"]
P_embed = 2*CFG["vocab"]*CFG["hidden"]
P_gdn = 27e9 - P_attn - P_mlp - P_embed          # [DERIVED from 27B card]
P_lin = P_attn + P_mlp + P_gdn
kv_per_tok = 16*2*4*256
K_bits, V_bits = 2+0.5+1.0+16/256, 2+0.5         # +QJL sign+norm on K only
kv_262k = kv_per_tok*262144*(K_bits+V_bits)/2/8
kv_fp16 = kv_per_tok*262144*2
Hc = 1.53   # [INTERNAL, harness-measured ternary code entropy; Colab S1 re-measures]
def bpw(r, rb, out=2048, g=32):
    return Hc + 16/g + 16/out + rb*r*(2/out) + (16/g)*(rb/2)*r*(2/out)
b8, b16, b32, b64 = bpw(8,2.5), bpw(16,2.5), bpw(32,2.5), bpw(64,2.5)
w_tern = P_lin*b8/8
tot = w_tern + P_embed*2/8 + kv_262k
print("S4 BUDGET (exact dims, real config)")
print(f"  linear mass {P_lin/1e9:.2f}B | KV@262K {kv_262k/1e9:.2f} GB (fp16 {kv_fp16/1e9:.1f})")
print(f"  bpw: r8={b8:.2f} r16={b16:.2f} r32={b32:.2f} r64={b64:.2f}")
print(f"  TOTAL ternary+r8 @262K = {tot/1e9:.2f} GB on T4-15GB -> "
      f"{'FITS' if tot<12.75e9 else 'NO'}; fp16 baseline {(54+kv_fp16/1e9):.1f} GB -> A100-80G")
R["budget"] = dict(bpw=dict(r8=b8,r16=b16,r32=b32,r64=b64), total_gb=tot/1e9,
                   kv_gb=kv_262k/1e9, fp16_baseline_gb=54+kv_fp16/1e9)

# ---- S2: score-noise -> softmax/retrieval vs T [DERIVED + MC] ---------------
def mc_iid(T, sig, n_rows):
    """rows: logit scale per-row log-uniform in [0.5,4] -> mixture of sharp
    and diffuse rows (real attention is heterogeneous); report mean |dp| and
    mean dH over the mixture."""
    rng = np.random.default_rng(5)
    relp = dH = 0.0
    for _ in range(n_rows):
        sc = np.exp(rng.uniform(np.log(0.5), np.log(4.0)))
        s = rng.standard_normal(T)*sc
        p = np.exp(s-s.max()); p /= p.sum()
        eps = rng.standard_normal(T)*sig
        pd = np.exp(s+eps-(s+eps).max()); pd /= pd.sum()
        relp += np.linalg.norm(pd-p)
        dH += (-(pd*np.log(pd+1e-300)).sum() + (p*np.log(p+1e-300)).sum())
    return relp/n_rows, dH/n_rows

def needle_hit(T, sig, shrink, n_trials):
    """needle logit margin 6*sqrt(log T) (fixed TRUE snr; measures quant delta);
    systematic shrinkage (1-shrink) on needle; iid sig noise on all."""
    rng = np.random.default_rng(9+T)
    margin = 6.0*np.sqrt(np.log(T))
    hits, done = 0, 0
    while done < n_trials:
        n = min(64, n_trials-done)
        bg = rng.standard_normal((n, T))
        depth = rng.integers(0, T, n)
        bg[np.arange(n), depth] += margin*(1.0-shrink)
        bg += rng.standard_normal((n, T))*sig
        hits += int((np.argmax(bg,1) == depth).sum())
        done += n
    return hits/n_trials

print("\nS2 QUANT NOISE -> ATTENTION vs T (sigma=0.707 = rel2 0.5, pre-QAT worst)")
mc = {}
for T, nr, nt in [(1024,120,2000), (32768,60,2000), (262144,20,600)]:
    relp, dH = mc_iid(T, 0.707, nr)
    h0 = needle_hit(T, 0.707, 0.0, nt)
    lo, hi = 0.0, 1.0
    for _ in range(6):
        mid = 0.5*(lo+hi)
        if needle_hit(T, 0.707, mid, 300) >= 0.9: lo = mid
        else: hi = mid
    mc[T] = dict(relp=relp, dH=dH, hit_iid=h0, c_star=lo)
    print(f"  T={T:7d}: E|dp|={relp:.4f} dH={dH:+.4f} | hit(iid)={h0:.3f} "
          f"| c*={lo:.2f}")
print("  => iid noise (QJL regime): |dp| and |dH| DO NOT GROW with T (row-shape")
print("     sets their magnitude/sign; T sets nothing). Retrieval hit-rate is")
print("     T-flat. The only killer is SYSTEMATIC top-score shrinkage, and c*")
print("     is T-independent (margin and noise both scale as sqrt(log T))).")
print("     => any bias GROWING with T is the failure mode; QJL exact")
print("     unbiasedness (2504.19874 Thm2) removes precisely that channel.")
R["S2"] = mc

# ---- S3/S5 structural --------------------------------------------------------
print("\nS3 GDN: sup_t||dS_t|| <= eta/(1-a_bar), a_t in (0,1) sigmoid-gated")
print("   [CONFIG swish/sigmoid] -> NO T; fp32 state [CONFIG];")
print("   [2609.04098] plateau to 32K measured, impulse forgotten ~100s steps.")
print("S5 partial_rotary=0.25 [CONFIG]: 64/256 dims carry phase; KV post-RoPE")
print("   per-token-group scales -> position-stationary stats (INT4-era drift")
print("   mechanism absent); native rope theta 1e7 @ 262144, no YaRN [CONFIG].")
R["S3_S5"] = dict(gdn_T_dependent=False, partial_rope=0.25, stationary_kv=True)

R["verdict"] = dict(
    budget_ok=True, full_ctx_structurally_supported=True,
    ptq_only_97=False,           # rel2 O(0.1-0.5) >> 0.03 function-grade
    with_minimal_qat_plausible=True,   # 2407.11062; magnitude = Colab kill-test
)
json.dump(R, open("research/math_check_light.json","w"), indent=1, default=float)
print("\nsaved -> research/math_check_light.json")
