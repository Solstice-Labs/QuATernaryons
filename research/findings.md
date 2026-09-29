# IsoQuant-Extreme — Research Evidence Log (persisted)

Verified sources via web search (Serper/Google), 2026-09-23. Part 1 of 2.

## A. Sub-2-bit weight quantization SOTA

1. **BitNet b1.58 2B4T** — arXiv:2504.12285 (Apr 2025). First open-source native 1-bit LLM at 2B scale, 4T training tokens. W1.58A8: ternary weights {-1,0,+1} ≈ 1.58 bits, 8-bit activations. Requires specialized kernels (standard libraries lack mixed-precision low-bit support). Official inference framework: github.com/microsoft/BitNet. Earlier JMLR v26 (24-2050) "BitNet: 1-bit Pre-training for LLMs" introduced BitLinear as drop-in nn.Linear replacement, straight-through estimator, shadow weights. HF blog "Fine-tuning LLMs to 1.58bit: extreme quantization made easy" (Sep 2024) covers post-training ternarization.
2. **AQLM** — arXiv:2401.06118 (ICML 2024, Egiazarian et al., cited ~280+). Additive quantization with learned multi-codebook lookup tables + beam-search coding, fine-tuned. Llama-2-7B @ 2-bit: WikiText2 PPL 6.93 (vs 5.21 fp16? — note: 5.21 is the fp16-ish delta reference; 2-bit 7B → 6.93). "First scheme Pareto-optimal below 3 bits." Code: github.com/vahe1994/AQLM. VPTQ (EMNLP 2024) is a related vector-PTQ baseline.
3. **QuIP#** — arXiv:2402.04396 (ICML 2024, Tseng et al., cited ~429). Three techniques: (i) randomized Hadamard incoherence processing (weight + Hessian-aware), (ii) E8 lattice codebooks, (iii) fine-tuning. Llama-2 70B @ 3-bit: WikiText2 PPL 8.22; 2-bit comparable to OmniQuant 3-bit. Code: Cornell-RelaxML/quip-sharp. Predecessor QuIP: "2-bit quantization of LLMs with guarantees" (LDLQ, incoherence).
4. **QTIP** — arXiv:2406.11235 (NeurIPS 2024, Tseng et al., cited ~96). Trellis-coded quantization (TCQ): stateful decoder → ultra-high-dimensional effective codebooks (e.g., effectively 2^(4×64) codewords at ~4-bit), bitshift trellis for hardware. Incoherence processing to make weights ~i.i.d. Gaussian, then VQ with TCQ; also LUT-only / computed-lookup-free variants. SOTA over QuIP# at 2-4 bits. Code: Cornell-RelaxML/qtip. Together.ai blog: "Even Better, Even Faster Quantized LLMs with QTIP."

## B. Rotation / incoherence methods (the dual-side "R W R^T" family)

1. **QuaRot** — (arXiv:2404.00456, NeurIPS 2024, Ashkboos et al.) "Outlier-Free 4-Bit Inference in Rotated LLMs": fixed random Hadamard rotations R1..R4 inserted into model; computational invariance y = (R W R^T)(R x)... makes activations outlier-free, fused online via FWHT.
2. **SpinQuant** — arXiv:2405.16406 (Liu et al., cited ~575). Learned rotations (Riemannian optimization on Stiefel manifold) beat Hadamard; same R1–R4 insertion points as QuaRot. Code: facebookresearch/SpinQuant.
3. **QuaRot/SpinQuant lineage in KV**: R3/R4 rotations act on attention inputs (K/V side) — the key precedent for dual-sided KV rotations.
4. **DuQuant** — (arXiv:2406.01721, NeurIPS 2024) rotation + permutation to flatten activation outliers for both weights and activations.
5. **GSR (Grouped Sequency-arranged Rotation)** — ACL 2025 SRW (Choi et al.): training-free improved rotation via sequency ordering, cited ~4.
6. **RotateKV** — arXiv:2501.16383 (Su et al., IJCAI 2025, cited ~40). THE closest prior work on the KV side. Outlier-aware adaptive rotations via channel reordering preserving FWHT efficiency; attention-sink-aware quantization (protect massive activations); <0.3 PPL degradation at 2-bit on WikiText-2 (LLaMA-2-13B), <1.7% GSM8K degradation. Code: github.com/ZunhaiSu/RotateKV.
7. **OSCAR** — arXiv:2605.17757 (May 2026). "Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization." Fixed rotations chosen offline from spectral/covariance structure. Project page: oscar-quantize.github.io. Also cites "Output-Aware Rotation for INT2 KV-Cache Quantization" (arXiv:2608.02691, Aug 2026).
8. AMD Quark docs confirm QuaRot R1–R4 taxonomy as industry-standard reference.

## C. KV cache quantization

1. **KIVI** — arXiv:2402.02750 (ICML 2024, Liu et al.). Tuning-free 2-bit KV: keys per-channel, values per-token (empirical distribution insight); 2.6× less peak memory including weights; Llama/Falcon/Mistral. Code: github.com/jy-yuan/KIVI.
2. **KV Quant (GPTQ-style, Hooper et al.)** — "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization" (NeurIPS 2024 poster 96936): non-uniform (NuQ) quantization, per-channel keys, pre-KV attention outliers, 4-bit → 10M context.
3. **KVTuner** — ICML 2025 (Li et al., PMLR v267): sensitivity-aware layer-wise mixed-precision; nearly lossless 3.25-bit (Llama-3.1-8B-Instruct) / 4.0-bit KV.
4. **TurboQuant** — vLLM blog (May 2026) study vs BF16/FP8 on long-context/reasoning workloads (4-bit KV).
5. **Survey/infra**: NVIDIA research blog "KV Cache Compression and Its Infra Problems" (Jun 2026); awesome-kv-cache-management (TreeAI-Lab).
6. Head-dim geometry: K head_dim=128 typically → per-head rotations are 128×128 (block-diagonal across heads); outlier channels in K are sparse and massive-activation-like (attention sinks, cf. "Massive Activations in LLMs", Sun et al. 2024).

## D. Geometry facts (to be verified numerically in verify_isoquant_math.py)

1. 4D rotation: any R ∈ SO(4) factors as left/right quaternion action: R(x) = L x R* with unit quaternions L, R̃ (double cover SO(4) ≅ (Sp(1)×Sp(1))/{±(1,1)}). Wikipedia "Rotations in 4-dimensional Euclidean space"; Stanford lecture notes; Reed blog "Why Do Quaternions Double-Cover?"
2. Isoclinic rotations: left (L=I) or right (R=I) 4D rotations = two equal rotation angles in two orthogonal planes; isoclinic matrices form subgroups (left/right-isoclinic SO(4) subgroups each ≅ SO(3)/S^3-parametrized); product of two right-isoclinic is right-isoclinic. Cayley factorization: any SO(4) = product of left- and right-isoclinic (Thomas, AGACSE 2015 "On Cayley's Factorization of 4D Rotations"; Cripps et al. Birmingham).
3. Isoclinic special property: all 2-planes rotate by the SAME angle θ → the 4D map acts as complex multiplication by unit quaternion on H ≅ R^4. Determinant 1, eigenvalues e^{±iθ} each with multiplicity 2.
4. Quaternions as pairs: unit quaternion ↔ SU(2); q acting by conjugation on imaginary part = 3D rotation. 4D isoclinic rotor: x ↦ q x (left mult) or x ↦ x q* (right mult).
5. Hadamard connection: for n divisible by 4, the n/n-normalized Walsh-Hadamard transform H_n is orthogonal; H_4 = (1/2)[[1,1,1,1],[1,-1,1,-1],[1,1,-1,-1],[1,-1,-1,1]] is a self-inverse orthogonal matrix; can be written as (up to sign/permutation conventions) an isoclinic-type composition. QuIP# uses H_n as incoherence rotation; QuaRot/SpinQuant likewise.
6. Clifford group equivariant networks (Ruhe et al., NeurIPS 2023; CG-EGNN arXiv:2410.04692; GCANs ICML 2023; GLGENN PMLR v267) = precedent for rotors in ML, but NOT for quantization — novelty of IsoQuant-Extreme is rotor-parametrized *dual-sided* rotation for quantization, not rotors in networks.

## E. Novelty positioning (IsoQuant-Extreme vs prior work)

- Prior art uses arbitrary orthogonal rotations (Hadamard fixed / learned Stiefel / covariance-aware offline). IsoQuant-Extreme claims: parametrize BOTH the weight-side and KV-side rotations as 4D isoclinic Clifford rotors applied per 4-channel block (block-diagonal orthogonal, block = left/right quaternion action), giving (a) 4-param-per-6-DOF compact parametrization with exact orthogonality by construction, (b) trivially invertible on-device (conjugate rotor, 4 params), (c) FWHT-compatible block structure for fused kernels, (d) joint weight+KV optimization ("dual-sided").
- Verified: no prior work combines quaternion/rotor parametrization with sub-2-bit weight or 2-bit KV quantization (search D returned no such work). Closest: RotateKV (adaptive rotation but Hadamard-structured, not rotor-parametrized), OSCAR (spectral fixed), QuIP#/QTIP (weight-only, Hadamard incoherence).
- MUST be stated honestly in paper: empirical claim is parity-or-better vs Hadamard at equal overhead; theory gives exact isometry guarantees (verified numerically) — no claimed transcendence of rate-distortion bounds.

## F. Kernel/hardware systems background

1. **Marlin** (IST-DASLab, arXiv:2408.11743; Frantar et al.): mixed-precision FP16×INT4 CUDA matmul, near-ideal 3.87× speedup up to batch ~16-32; global-memory weight layout with tensor-core-friendly ldmatrix dequant. Key design: weight repacking for coalescing + warp-level shuffle-based dequant.
2. **Machete** (Neural Magic, Hopper): mixed-input GEMM via wgmma, dual reference (read fp16 activation, dequant weight in-register).
3. **QServe** (MIT/SJTU, arXiv:2405.04532): W4A8KV4 with smooth attention (note: KV4 precedent for low-bit KV serving).
4. **BitNet kernels**: Microsoft BitNet framework (custom MSL/CUDA for ternary matmul, I8_S8_F32 accumulate).
5. **Metal**: MSL simdgroup matrix ops (simdgroup_matrix, simdgroup_multiply_accumulate), threadgroup memory staging, float4 vector types (natural fit for 4-channel rotor blocks); Apple ANE unavailable for custom kernels → GPU path.
6. **Triton**: tl.load/tl.dot with int accumulation, block-pointer style; dequant-in-kernel pattern (e.g., GPTQ-Marlin triton ports); 2-bit unpack via bit-shift masks.

## G. Numerical/theory scaffolding for the paper (design decisions, to verify numerically)

- Rotor block B(q) ∈ SO(4): B[q_L][i,j] via x ↦ q ⊗ x (left-isoclinic): parameter count 4 (unit quaternion) per 4×4 block (vs 6 for Givens full block, 16 for free orthogonal) — orthogonality exact by norm constraint ‖q‖=1, preserved under reparameterization q = v/‖v‖.
- Dual-sided identity: y = (Q_K ⊗_blk K) requant; logits use (Q_K^{-1} Q_V^{-1}) identity fused into attention — actually exact factorization: attention scores s = k^T q_mask... = (Rk)^T (Rq) since R^T R = I (invariance lemma). For V: o = A (R v) → apply R^T after attention (with sink protection from RotateKV literature).
- Quantizer: per-group 2-bit asymmetric affine (4 levels, e.g., levels {0,1,2,3} → scaled) for KV; for weights sub-2-bit: E8-lattice or TCQ codebook interchangeable (cite QuIP#/QTIP); IsoQuant contribution orthogonal to codebook choice (ablation axis).
- Error bounds: dithered/uniform quantization error model; incoherence bound ‖W − Q̂‖_F under RWR^T with random orthogonal R (cite QuIP theory, LDLQ); isoclinic rotor family ⊂ random-orthogonal family → same incoherence lemma applies with block-constant structure; verify Frobenius error numerically vs Hadamard baseline on synthetic Gaussian + outlier-spiked matrices.

## H. VERIFIED NUMERICAL RESULTS (verify_isoquant_math.py, seed 20260923, numpy 2.4.6, Python 3.12) — ALL 21 CHECKS PASS

Final calibrated scheme: 6-stage rotor butterfly, strides [8,4,8,4,8,4] (coarse-first), per-block Cayley double rotors A(L)B(R) (8 params/block/stage) fitted to moment EQUALIZER targets U*_j = V_j H4_half, 1 coordinate-descent sweep, last-stage MSE-aware refinement (near-identity SO(4) perturbations, 48 candidates/block, 2 passes).

Table 1 (rel. Frobenius reconstruction error, 2-bit group-32, d=128):
- iid:          identity 0.4549  Hadamard 0.4546  rotor(plain) 0.4534  butterfly(rand) 0.4503  butterfly(calibrated) 0.4073
- spiked(4ch x10): identity 1.1049  Hadamard 0.4070  rotor(plain) 1.0939  butterfly(rand) 0.4812  butterfly(calibrated) 0.3864
- anisotropic:  identity 0.4578  Hadamard 0.4522  rotor(plain) 0.4482  butterfly(rand) 0.4506  butterfly(calibrated) 0.4046
5-seed anisotropic means: identity 0.4556, Hadamard 0.4544, calibrated butterfly 0.3998.

Table 2 (dual-sided 2-bit KV, T=256, d=128, K spikes x50 on 3 ch, rotor calibrated on keys):
- output rel. error: identity 0.9228 vs calibrated-rotor 0.7410 (x1.25)
- logit rel. error:  identity 0.2253 vs calibrated-rotor 0.0751 (x3.00)

Machine-precision identities verified: A(L),B(R) in SO(4); homomorphisms; commutation; isoclinic spectrum; Cayley factorization of random SO(4) to 7e-16; Procrustes closed form optimality; ||S(G)||/2 <= ||G||_F; score invariance (Rk)^T(Rq)=k^Tq; value path exactness; softmax row-stochastic ||A eps|| <= sqrt(T)||eps||.

Negative results (documented in paper as ablations):
- Plain block-diagonal rotors CANNOT disperse spikes confined to one 4-block (1.0939 vs identity 1.1049) — motivates permutation stages.
- PCA-eigenbasis Procrustes fit is the WRONG target (spike subspace invariant); moment-equalizer target required.
- Fine-first stride schedule [4,8,...] stalls dilution vs coarse-first [8,4,...] (0.4587 vs 0.4435 pre-refinement).

## I. PRIOR-ART CORRECTION (2026-09-23, session 2) — CRITICAL

User flagged prior work. VERIFIED by reading the paper:
- **IsoQuant** = Ji, Zhongping, arXiv:2603.28430 (Mar 2026), code ParaMind2025/isoquant.
  Blockwise 4D isoclinic quaternion pairs T(v)=qL v qR_bar for KV-cache stage-1
  online VQ (TurboQuant framing). Variants: Full (6-DOF SO(4), 1024 FMA @ d=128),
  Fast (single isoclinic factor, 512), 2D Givens aux (256). Fused CUDA vs
  RotorQuant: mean 4.49x/4.66x speedups, 18 settings, MSE parity. LIMITATION
  (their own words): "validation is limited to the stage-1 quantize-dequantize
  path on synthetic normalized vectors; full end-to-end KV-cache evaluation on
  real model activations remains future work." Random Haar quaternions or
  unconstrained normalization; NO fitting/calibration theory.
- **RotorQuant** = Pope, John D., 2026, scrya.com/rotorquant — Cl(3,0) 3D rotors
  (42 blocks + 2D tail at d=128; ~2408 FMA), the predecessor Ji improves on.
- **TurboQuant** = Google Research blog (2025): dense random rotation + Lloyd-Max
  + QJL residual correction; the online-VQ framing both build on.
- **HQMQ** = "Hurwitz Quaternion Multiplicative Quantization" arXiv:2605.27646
  (May 2026): claims to beat IsoQuant/RotorQuant preconditioning on outlier-heavy
  models (Hurwitz lattice codebook, scalar Lloyd-Max).
- Lineage: TurboQuant(dense) -> RotorQuant(Cl3) -> IsoQuant(SO4 isoclinic) -> HQMQ.
- OUR PAPER REPOSITIONED: novelty = (a) weight side sub-2-bit, (b) MULTI-STAGE
  permutation-rotor global mixing (single-stage blocks CANNOT disperse spikes —
  our verified ablation 1.0939 vs 1.1049; invisible in IsoQuant's i.i.d. eval),
  (c) closed-form Rotor-Procrustes + Cayley-coverage calibration theory,
  (d) attention perturbation bound, (e) MSL/Triton specs. Ji cited as strictly
  prior + independent on the shared block-rotor idea. references.bib + main.tex
  updated (related work rewritten, intro corrected).

## J. Zero-training near-lossless landscape (verified this session)
- Unsloth Dynamic 3.0 GGUFs (Aug 2026): sensitivity-aware mixed-precision GGUF,
  zero training, near-lossless at ~4-6 effective bpw; the practical mixed-precision lever.
- Bonsai 2 27B (PrismML, Sep 2026): ternary, 98.2% of Qwen3.8 27B benchmarks,
  5.9GB — but it is QAT (trained-for-ternary), NOT pure conversion. Proves the
  training branch closes sub-2-bit; does not contradict the PTQ gap.
- Google TurboQuant (2025): zero-training near-lossless KV at ~4-bit-class with
  residual correction; 6x memory claims from secondary coverage.

## I. REAL-WEIGHTS RUN (Bonsai-27B bf16 GGUF, 2026-09-23)
- Script: quantize_real.py (mmap streaming, seeded, per-head k/v, cols-side o)
- 12 tensors (layers 0/2/3/5), 2-bit g=32, relative Frobenius error:
  * Means ALL:    identity 0.4849 | Hadamard 0.4549 | rotor(rand) 0.4566 | rotor(cal) 0.4512
  * Means k/v:    identity 0.4909 | Hadamard 0.4547 | rotor(rand) 0.4556 | rotor(cal) 0.4486
  * Per-head k:   cal beats H on all 4 layers (best 0.4389 vs 0.4541, blk0)
  * attn_o d=5120: H slightly best (0.4553 vs cal 0.4565) - consistent with
    Prop. incoherence (ii) dilution gap at large d; honest negative reported.
- Identity worst in 10/12 tensors; two exceptions = deep-layer v/o (near-isotropic).
- Conversion cost: ~3 min single-threaded M4 (numpy), no GPU, no training.
- Paper updated: Table tab:real + Findings (4) + Limitations (2) + Conclusion.

## J. PERFORMANCE FIXES (verify_isoquant_math.py, no semantic change)
- refine_last_stage_mse: full d x d R.copy() per candidate -> in-place block
  update with exact G^T revert. d=5120 attn_o: >10 min -> 43 s (14x+).
- rotor_butterfly fit loop: per-stage prefix rebuild (30 matmuls) -> incremental
  prefix advance (6 matmuls).
- gauss_newton_factorize: early exit at residual <= 1e-13.
- Suite re-verified after changes: 22/22 PASS. CLI: --fit-rows (calibration
  subsampling; error eval always exact), --kinds, --refine-cands wired through.

## K. DEEP 2026 LANDSCAPE SWEEP (parallel-cli advanced mode, 2026-09-23)

### K1. WEIGHTS 2-BIT: NEW 2026 METHODS WE WERE MISSING
- TWLA (arXiv:2606.13054, ICML 2026): W1.58A4 via PTQ. KOTMS Kronecker
  orthogonal rotation reshapes weights to ternary-friendly tri-modal dists.
  Code: github.com/Kishon-zzx/TWLA. NEAREST NEW COMPETITOR on rotation-for-ternary.
- CAT-Q (arXiv:2606.26650): ternary PTQ w/ 512 calib samples, beats BitNet
  b1.58 v1/v2 (100B-token QAT) - claims 100,000x cheaper.
- PiSO (arXiv:2606.10890): closed-form optimal per-channel scales for RTN;
  benefits grow as bits narrow; composes w/ error correction.
- D2Quant (arXiv:2602.02546): sub-4-bit weight PTQ; dual-scale quantizer for
  down-projections + LayerNorm mean-shift correction; Qwen3-32B 64.56 vs
  GPTAQ 61.21.
- BTC-LLM (arXiv:2506.12040): sub-1-bit (0.7-1.11bpv) learnable (diag+orth)
  transform + binary codebook. Llama-2-7B 0.8-bit PPL 6.60 (STBLLM 13.06);
  1.11-bit 6.06 vs fp16 5.47 (13B). Claims it surpasses 2-bit VQ methods.
- KronQ (arXiv:2607.07964): Kronecker-factored Hessian, 2-bit Llama-3-70B.
- PTQTP (arXiv:2509.16989): trit-plane decomposition (binary x ternary).
- HIGGS (NAACL 2025): Hadamard + MSE-optimal grids, data-free, beats NF4.
- BITCOS (HN Sep 2026, "Breaking 1.58-bit Barrier"): zeros are up to 51.5%
  of ternary weights -> distribution-adaptive layout -> 1.48 bpw.
- Also: SliM-LLM, LittleBit (Feb 2026), ParoQuant (Feb 2026, pairwise
  rotation), GlowQ (Apr 2026), SliderQuant (2026), ARB-LLM, STBLLM.

### K2. VERIFIED 2-BIT WEIGHT PPL ANCHORS (published numbers)
- Llama-2-7B fp16 = 5.12 (PV-Tuning, 2405.14852, Table 9)
- QTIP 2-bit Llama-3-8B = 7.33 (WikiText2; cited in arXiv:2608.30076 Table 1)
- QTIP 4-bit Llama-3-8B = 5.67; AWQ W4 Llama-3.1-8B = 7.00 (same table)
- QuIP# 2-bit Llama-3-8B-Instruct = 11.23 (same table, instruct caveat)
- DB-LLM 2.01-bit Llama-2-7B = 7.23; OneBit 1.01-bit = 9.73;
  BiLLM 1.08-bit = 32.48 (PV-Tuning Table)
- BiLLM 1.08-bit Llama-2-70B = 8.41; Llama-65B = 8.49
- AQLM ~1-bit (1x8) Llama-2-7B = 7.85 (finetuned; AQLM repo 2025.04)
- BTC-LLM 0.8-bit Llama-2-7B = 6.60 (alphaXiv summary)
- PV-Tuning: first Pareto-optimal 2-bit Llama-2 family.

### K3. KV 2026 COHORT (context for related work)
- SAW-INT4, Provable Randomized-Hadamard Quantization, OScaR, OCTOPUS
  (octahedral parametrization) - all 2026 KV papers (via emergentmind).
- TurboQuant vLLM merged May 2026: 4bit_nc within ~1pt of FP8; 3bit_nc
  drops 15-25pt on AIME25/LCB; paper config (3.5bpv+QJL) "identical FP16
  on LongBench per paper, worse in practice" (OnlyTerp README - skeptical).
- RotorQuant (Pope): "better PPL, 28% faster decode, 5x faster prefill,
  44x fewer params" (scrya-com/rotorquant). Community impl:
  abysslover/rotorquant_improved = Turbo+Rotor+Iso+Planar unified.
- IsoQuant GitHub: github.com/ParaMind2025/isoquant (Ji, Mar 2026).
- CARE/QuatRo (arXiv:2511.11665, NeurIPS'25 UniReps WS): Clifford rotors
  for RoPE - adjacent area, not a competitor.

### K4. IMPLICATIONS FOR OUR PAPER
1. 2-bit weights, TRAINING-FREE, published SOTA = QTIP 7.33 on Llama-3-8B
   (~+1.6 over fp16) => "near-lossless 2-bit weights training-free" is NOT
   solved by anyone; our earlier prediction (base+1..3) is confirmed.
2. Novelty watch: TWLA (rotation-for-ternary, Kronecker) + ParoQuant
   (pairwise rotation) are new; NEITHER uses 4D isoclinic rotor butterflies
   or closed-form rotor calibration. Our delta stands but related work
   must cite TWLA/CAT-Q/PiSO/D2Quant/BTC-LLM/KronQ/HIGGS/BITCOS + 2026 KV
   cohort (OScaR/OCTOPUS/SAW-INT4).
3. Our strongest remaining claim: transform slot on top of QTIP/TCQ
   (swap Hadamard incoherence -> calibrated rotor butterfly). SpinQuant
   precedent: learned-vs-random matters MOST on hard models at low bits.
   Predicted gain: 0.05-0.3 PPL at 2-bit. That is the Colab experiment.

## L. NEW-GEN MODEL SWEEP (Qwen3.8 / Gemma 4 / DeepSeek-V4.1-Flash, 2026-09-23)

### L1. QWEN3.8-27B — MEASURED ANCHORS (the model we target!)
- Base PPL MEASURED by Unsloth KLD/PPL eval: PPL(base) = 6.9505 +/- 0.045
  (HF discussion #74). Replaces our 5.0-6.0 guess. <3% degradation target
  => PPL <= 7.16 (delta <= +0.21).
- Architecture (unsloth card): 64 layers HYBRID = 48 Gated DeltaNet (linear
  attn, no KV cache) + gated attention (24 Q / 4 KV heads, head_dim 256,
  RoPE 64), FFN 17408, vocab 248320. KV cache only in ~1/4 of layers.
- Unsloth Dynamic V3.0 (Aug 19): "+10% top-1% acc at same size"; 4-bit
  UD-IQ4_XS PPL ratio 1.0078 (+0.054) and 1.0027 (+0.019) => measured 4-bit
  delta on this model = +0.02..0.05 PPL (<1%).
- 2-BIT CLIFF confirmed: "sharp drop from UD-Q2_K_XL to UD-IQ2_S" (Div-300
  ~25% -> <8-10%). Quesma task benchmark: "4-bit holds up, 1-bit collapses".
- Minima (Sep 4): ALL 496 linear layers (incl. 48 GDN blocks!) to NVFP4
  W4A4 => matches BF16 within seed noise (0.52 pts 5-task avg; AIME 26/30
  on all seeds). GDN gate projections LEAST sensitive (2.1-2.6% output
  movement) - contradicts the everyone-protects-GDN-in-FP8 convention.
- NVFP4 checkpoints: nvidia/Qwen3.8-27B-NVFP4, RadixArk (ModelOpt W4A4).

### L2. GEMMA 4 (Apr 2, 2026; QAT Jun 5, 2026)
- Sizes E2B/E4B/12B/26B-A4B/31B (+ MoE variants). QAT checkpoints for
  Q4_0 + novel mobile format: E2B < 1GB; QAT cut Q4_0 PPL drop 54% vs PTQ
  (Gemma 3 data). Edge variants: last-N layers REUSE KV tensors from
  earlier layers; global attn GQA 16->4 KV heads @ head_dim 512, K=V
  (attention_k_eq_v). Unsloth UD ladder covers it (UD-Q2_K_XL ~10.5GB/26B).

### L3. DEEPSEEK-V4.1-FLASH (Sep 10-18, 2026) — DIRECTLY RELEVANT
- 552B backbone + 196B ENGRAM params (748B total), 8B/16B active, 1M ctx.
- Engram = conditional memory: modernized n-gram embeddings, O(1) lookup,
  host-offloaded tables (github.com/deepseek-ai/Engram, 4.7k stars).
- DSpark = their speculative decoding arch (explains "dspark" in Bonsai
  GGUF names we quantized!).
- PRODUCTION FP4 MAIN KV CACHE: "quantize the cache AFTER RoPE: quantizing
  before RoPE yields only marginal improvement + decode overhead; FP8 for
  SWA KV due to sensitivity." => (a) frontier labs now ship FP4 KV;
  (b) their RoPE-ordering ablation matches our rotate-after-RoPE exactness
  check; (c) mixed-precision KV (FP8 SWA) in production = KVTuner-style.

### L4. FORMATS & METHODS (late-2026)
- TurboQuant FOR WEIGHTS (community ports): random rotation + Lloyd-Max
  on weights, near-optimal 4-bit + lossless 8-bit residual, 3.2x savings.
- Proteus (ICML 2026): lookup-free TCQ for 2-bit weights (MUL-BAL codes).
- SignRoundV2 (arXiv:2512.04746): mixed MXFP4/6, ~1% gap @ 4.5 avg bits;
  "substantially improving 2-bit weight-only" via grad+recon bit allocation.
- QTEA (EMNLP 2026, arXiv:2609.00224): ternary 1.7bpw effective + salient
  residual; Qwen3-14B +16.7% acc vs best ternary PTQ.
- "Below Two Bits" survey (OpenReview): 23 methods 2023-2026, 4-axis
  taxonomy incl. late-2026 calibration-only ternary wave.
- Sherry (ACL 2026): 3:4 sparsity => 1.25-bit ternary packing; QAT lane.

### L5. IMPLICATIONS FOR OUR PLAN (<3% target)
1. Anchor now MEASURED: base 6.9505; 4-bit = +0.02..0.05. <3% => PPL<=7.16.
2. Flat 2-bit will NOT reach <3% (2-bit cliff on this very model is
   documented; QTIP-class +29% on Llama-3). Honest verdict: <3% needs
   sensitivity-driven mixed precision landing at ~3.0-3.5 avg bpw; the
   EXPERIMENT is measuring the bpw-vs-PPL curve and whether calibrated
   rotors shift it left vs Hadamard at every point.
3. Minima's GDN finding says our allocator will find most GDN linears
   cheap (2-bit OK) and concentrate bits in attention/MLP-down/gates.
4. DeepSeek FP4-KV + RoPE-ordering = production validation of our KV lane;
   cite in related work + positioning.

## §M. QAT-polish pivot + long-context quantization (2026-09-24)
Sources: EfficientQAT (arXiv 2407.11062, ACL 2025 Best-resource candidate; github OpenGVLab/EfficientQAT), PEQA, QA-LoRA, TerDiT (2405.14854), "Does quantization affect models' performance on long-context tasks?" (arXiv 2505.20276, EMNLP 2025), MILLION (2504.03661), MoQAE (ACL 2025), Ternary Mamba (2606.18114), Scaling Law for QAT (2505.14302).

### M.1 EfficientQAT recipe (the "tiny QAT" template)
- Two phases: Block-AP = block-wise training of ALL params against cached block input/outputs (block outputs from the fp16 frozen model; only one block trains at a time -> memory = model + 1 block of grads + cached IO). E2E-QP = end-to-end training of ONLY quantization params (step sizes), quantized backbone frozen. Trainable count tiny (scales only).
- Results: 2-bit Llama-2-70B on ONE A100-80GB in 41h, <3% acc deg (69.48 vs 72.41); W2g64 Mistral-Large-123B -> 35GB with 4-pt deg. Uniform-int formats, T-MAC kernel support.
- Q-PEFT (QLoRA etc.) merging problem: merging LoRA reverts to fp16 -> re-PTQ needed; QA-LoRA merges into zero points instead; PEQA trains step sizes only (RTN init, suboptimal init hurts).
- Scaling Law for QAT (2505.14302): guidance for tokens/steps vs model size.
- TerDiT lesson: "conversion or post-training quantization from trained models does not help" at extreme low-bit WITHOUT a QAT component (their pilot: slow convergence, large activations) -> QAT component is mandatory for ternary-class fidelity.

### M.2 Long-context + quantization (critical for 262k target)
- 2505.20276 (EMNLP 2025), 9.7K tests, 5 models 7B-72B x 5 methods: 8-bit ~lossless (FP8 -0.2%, GPTQ-int8 -0.8%); 4-bit substantial: GPTQ-int4 -2.7% avg, AWQ -1.8%, BNB-nf4 -6.9% (up to -59% ONERULER Llama-3.1-70B). Retrieval degradation GROWS with input length (up to -23% avg at 128K for 4-bit). Strong model heterogeneity (Qwen-2.5-72B robust under nf4; Llama-3.1-70B -32% same task). Non-English up to 5x worse.
- Hypothesized mechanism for long-ctx sinks: accumulated error in RoPE embeddings (Wang et al. 2025a) -> motivates RoPE-aware mixed precision: q_proj/k_proj (pre-RoPE path) get larger bit/residual budget than v/o.
- No published sub-2-bit long-ctx results at 128K+ exist -> claim ceiling: short-ctx ~3% deg (published SOTA), long-ctx validated empirically at 32K->262K on Qwen3.8-27B (Colab) with RULER-style probes + PPL on long windows.
- KV side: MILLION (4-bit product-quantized KV, trivial PPL loss, 2.09x speedup @32K), MoQAE (mixed-precision KV for long ctx), KVC-Q (dynamic heterogeneous KV) -> dual-sided paper story stays current; KV long-ctx quant is hot.

### M.3 Design decision (user directive 2026-09-24)
Pivot from pure-PTQ to PTQ-init + tiny QAT polish: block-wise cached-IO training of quant params (per-group alpha/beta, per-column rescale, rank-r factors; T codes frozen; R frozen) + short E2E pass on MIXED-length sequences (1024 + 8192) so quant params adapt to long-ctx activation stats. RoPE-aware mixed precision (q/k more budget). Target: <=2.2 bpw honest, ~97-99% fp16 retention short-ctx, graceful long-ctx. Resource budget: M4 local for 1.7B pilot (hours), Colab T4 for 27B (E2E-QP-style only, offloaded frozen weights).

## N. RENAME + LANDSCAPE UPDATE (2026-09-27, scrapling MCP session; UPDATED 2026-09-29)
- PAPER RENAMED (2026-09-27): "IsoQuant-Extreme" -> "QuaTernaryions"
  (user-coined portmanteau of quantization + ternary + quaternions; zero
  arXiv hits for the exact phrase, verified).
- **RENAME v2 (2026-09-29, user): "QuaTernaryions" -> "QuATernaryons"
  (capital Q-U-A; "ons" not "ions"). Applied to main.tex title + all 6
  method-name occurrences, references.bib header, vm_qtern.sh comment.
  NOTE: the user's HF demo repo is named `SolsticeAI/QuATernaryions-demo`
  (with "ions") — different spelling from the paper title
  ("QuATernaryons"). Flagged in HANDOFF §0; NOT renamed by us (repo owner
  decision, model repos are immutable-by-name). Original 2026-09-27
  record: "QuaTernaryions" (final,
  user-coined portmanteau of quantization + ternary + quaternions; zero
  arXiv hits for the exact phrase, verified). Method name in text:
  "QuaTernaryion". Rejected along the way: Ternion, "Ternions Are All
  You Need" (user: not a copycat), Ternion-Turn puns, Rotors Not
  Matrices, Quantize in Four Dimensions. Reason for rename: head-on
  name collision with Ji's IsoQuant (arXiv:2603.28430) muddied the
  novelty claim. 5 self-references updated in main.tex;
  ji2026isoquant/pope2026rotorquant citations untouched.
  Attributions VERIFIED via arXiv API: IsoQuant = Zhongping Ji (Mar 2026,
  improves on RotorQuant 2408->1024/512 FMA); RotorQuant = John D. Pope.
- Name vetting (arXiv API): Ternion/BiRotor/SpinCache clean in ML;
  QUIRE taken (posit accumulators). "Ternions are all you need" exact
  title clean; genre has quantization precedents (Block Rotation is All
  You Need for MXFP4, arXiv:2511.04214).
- NEW 2026 COMPETITORS FOUND (cited in paper now):
  * HARP (arXiv:2605.29843, Zagitov/Molodtsov/Beznosikov): learnable
    butterfly block-orthogonal two-sided rotations, RHT-initialized,
    clearest gains at 2-bit, 90% of RHT throughput. Nearest weight-side
    calibrated-transform competitor; differentiator = quaternion
    parametrization (8 floats/block) + closed-form Procrustes fit vs
    their learned sparse factors.
  * RotaryQuant (arXiv:2608.08081, Lui et al.): MoE serving system; KV
    axis = WHT + block-diagonal SO(4) rotations ("IsoQuant" name used
    INTERNALLY by them too) at 3-bit KV, Metal kernels, dPPL<=0.0012.
    Confirms block-rotor viability in production; not a fitting theory.
  * Great Inversion survey (arXiv:2608.25188, Jokar): transform-stage
    survey, 43 methods; framing citation for flatten-vs-concentrate.
  * KV-COBRA (arXiv:2609.24298): per-head bit-rank allocation under
    fused Hadamard; relevant to our mixed-precision allocator.
- Session tooling: scrapling MCP server (HTTP, port 8765, token in
  /tmp/scrapling-mcp-token); helper research/mcp_call.py. Raw search
  dumps in .just-scrape/. User rule: scrapling for web search, not
  built-in web_search.

## O. OrcaSAQ2-27B — 3-bit 27B anchor (2026-09-29, verified from README)

NOTE ON SCOPE: this section adds ONLY the OrcaSAQ2 data point. The Qwen3.8-27B
architecture facts (48 GDN + 16 full-attn hybrid, 24Q/4KV heads, head_dim 256,
partial_rotary_factor 0.25, GDN state fp32, mass split 1.17B attn / 17.11B mlp
/ 6.17B GDN, KV@262K = 3.25 GB) were ALREADY established in HANDOFF §9.2 on
2026-09-24 from the local snapshot — an earlier draft of this section
"re-discovered" them from the OrcaSAQ2 config.json. That was redundant; §9.2
remains the authority for Qwen3.8-27B internals.

**orcarouter/OrcaSAQ-2-27B** (Apache-2.0, Continuum-AI-Corp/OrcaRouter) is a
third-party production quantization of Qwen3.8-27B — the strongest published
27B anchor, and NOT sub-2-bit:

| metric | value | note |
|---|---|---|
| base | Qwen/Qwen3.8-27B, 54 GB BF16 → **12.3 GB** (77.2% smaller) | |
| decoder avg | **3.21 bpw** | proprietary "sensitivity-aware mixed-precision", method NOT disclosed |
| WikiText-2 PPL | 5.6468 → 5.6482 (**+0.02%**) | 16,376 predicted tokens |
| top-1 agreement vs BF16 | **93.2%** | same eval path, same weights |
| mean KLD | **0.031** | |
| context | 262K | matches §9.2 |
| serving | vLLM + custom OrcaSAQ2-kernel + MTP spec-decode (+38% single-stream, 90.1 tok/s on 16 GB) | requires their vLLM integration |

**What this changes for IsoQuant-Extreme:**
1. A first-class 2026 lab ships 27B at **3.21 bpw** and calls +0.02% PPL at
   that rate "high fidelity" (PPL 5.6468, consistent with the Unsloth 6.9505
   measurement difference being eval-protocol dependent — different token
   counts, same ballpark). Our ternary2 ~2.17 bpw target sits ~1 bpw below
   what anyone ships; the paper's honest comparison is "what rank-r budget
   closes the 3.21→2.17 gap and at what PPL cost".
2. Their eval protocol (WikiText-2 PPL + top-1 agreement + KLD vs BF16, same
   weights same path) is the right co-primary metric set — adopt top-1
   agreement + KLD alongside PPL so our table is directly comparable to
   theirs (and to §9.2's Qwen budget numbers).
3. Their "93.2% top-1 agreement" framing = ~7% token divergence at 3.21 bpw.
   If our ternary2+QAT holds >90% agreement at ~2.17 bpw on SmolLM2 first,
   that is a headline.
4. vLLM shipping implies custom packed CUDA kernels for mixed-precision 3-bit
   — same deployment gap our kernel-spec section addresses at 2-bit.

## P. IsoQuant (Ji) FULL READ — what the #1 "collision" paper actually does (2026-09-29)

Read arxiv 2603.28430v1 in full. Corrects/completes our §N and paper §related:

1. **IsoQuant is KV-CACHE-ONLY, weights never touched.** Its scope: stage-1
   (rotate -> Lloyd-Max scalar quant -> unrotate) of *per-head vectors* for
   online KV quantization. TurboQuant lineage (dense rotation + QJL residual
   stage-2). Our weight-side ternary work has NO overlap with their results.

2. **Their evaluation is synthetic-only.** Reconstruction MSE on *synthetic
   normalized i.i.d. Gaussian vectors* (batch 8192, d in {128,256,512},
   bits {2,3,4}) + fused CUDA kernel latency on an RTX 4090. Their own
   abstract: "full end-to-end KV-cache evaluation on real model activations
   remains future work." They never run a model. No PPL, no long-context,
   no real cache tensors, no RoPE interaction study.

3. **Their transform is per-4-block random-diagonal, NOT calibrated.**
   IsoQuant-Full: v_i -> qL_i v_i conj(qR_i) with RANDOM Haar unit
   quaternions (or unconstrained-normalized); they have no fitting theory
   for data. Our Procrustes/fitting (Lemma 3, Cayley covering, block
   equalizer targets) is exactly the piece they lack. Their own Sec 7
   probabilistic analysis (E[R Sigma R^T] = blockwise-isotropic in
   expectation) is the *justification for random* — it explicitly
   acknowledges cross-block correlation is only "scrambled", not removed.
   Our moment-equalizer + multi-stage permutation composition targets
   exactly what they admit is left on the table.

4. **Speed numbers (their Table 1/2, d=128):** dense TurboQuant rotation =
   16384 FMAs; RotorQuant (Cl(3,0), 3D blocks) = 43 blocks, ~2408 FMAs;
   IsoQuant-Full (2 quaternions/block) = 1024 FMAs, 256 params;
   IsoQuant-Fast (single isoclinic factor, SO(3) subgroup) = 512 FMAs,
   128 params; 2D planar special case = ~256 FMAs. Kernel speedups vs
   RotorQuant fused CUDA: mean 4.49x (Full) / 4.66x (Fast) / 4.66x (2D),
   peak >6x, at matched reconstruction MSE.

5. **The 4D-block hardware argument aligns with our rotor story:** d=128 =
   32 exact 4-blocks, no 3-block remainder/tail, float4 SIMD mapping,
   register-resident fused kernel. This is *why* 4D isoclinic blocks are
   the right grain — cite them for it.

6. **Positioning of QuATernaryon vs IsoQuant (final, for paper):**
   complementary axes, not rivals: they = KV-side stage-1 with random
   quaternion blocks, synthetic-only validation; we = weight-side sub-2-bit
   with CALIBRATED rotors + permutation composition + real-model PPL +
   (pending) KV eval at 2-bit with the same machinery. Their "future work"
   list is literally our current experimental program:
     - end-to-end KV eval on real activations (our planned QuantizedCache)
     - calibrated (vs random) quaternion fitting
     - cross-block correlation removal (our multi-stage composition)
   Our novelty claims (b)(c)(d)(e) survive intact; do NOT overstate "first"
   on 4D isoclinic blocks per se — Ji has that for the KV stage-1 kernel.
   Cite generously, differentiate on calibration theory + real-model scope.

7. **Action items:** (a) keep paper text as-is (already accurate, re-verified
   today); (b) our planned KV-cache quantization eval IS the differentiator —
   prioritize it after weights ladder lands; (c) cite Ji 2026 in any KV
   section as the synthetic-stage-1 predecessor.
