# HANDOFF — IsoQuant-Extreme: real-model quantization campaign

Write date: 2026-09-25. **Read §0 FIRST — it is the only state that is live.**
Sections §1–§9 are the accumulated record; §9 is a 2026-09-24 evening snapshot
and §3's compute notes are partly superseded by §0.

---

## 0. LIVE STATE — 2026-09-29 14:00 IST (Kaggle primary lane)

**Primary compute = KAGGLE now, not Colab.** Colab free T4s kept getting
reclaimed after ~1 h (three sessions lost: isoq2, isoq4 x2). Kaggle gives
30 h GPU/week, 12 h max session, 2x T4 — the whole 3.5-4 h pipeline fits in
one session.

### How to drive the Kaggle lane

```bash
bash colab/kaggle_push.sh                      # push code dataset + kernel
kaggle kernels status solsticeaiofficial/qtern-smol
python3 colab/kaggle_tail.py [owner/slug] [max-events] [tail-lines]
                                               # LIVE log while RUNNING
                                               # (SSE endpoint /api/v1/kernels/logs/stream/)
kaggle kernels output solsticeaiofficial/qtern-smol -p /tmp/kq_out   # after done
```

Kernel: https://www.kaggle.com/code/solsticeaiofficial/qtern-smol (private).
One-time web-UI prereqs: HF_TOKEN secret, internet enabled, phone verified.

### v3 run state (started 2026-09-29 ~08:25 UTC)

- PHASE0 math 22/22 EXIT=0, PHASE1 calib 96/96 EXIT=0 (~3.6 min)
- PHASE2 lead-arm quant (96 modules, ~70s each, ~85 min) — IN FLIGHT
- then: base+mixed evals, QAT (mixed->fallback fineweb), evals, uniform arm,
  QAT, evals, bpw report. Expect ~3.5-4 h total.
- v1/v2 postmortem: v1 dead (sources not copied from /kaggle/input — fixed
  with glob+copy); v2 dead (EVAL_MIXED KeyError 'S': ternary2 shards have
  no S — _apply_quant now uses `S if "S" in z else None`; fixed+repro'd
  locally). v3 also tails every subprocess log into the kernel log so PPL
  lines are visible live.

### CRITICAL fixes landed this session (2026-09-29)

1. `_apply_quant` (isoquant_weights.py ~L789): `S=z["S"] if "S" in z else
   None` — ternary2 shards carry A/B/U/V, not S. This bug killed every
   quantized eval on Kaggle v2 AND would kill any eval of quant_ternary2m
   anywhere. Repro + fix verified locally (APPLY_OK + forward OK).
2. `build_rank_map()` (isoquant_weights.py): budget-neutral rank allocation
   — assigns ranks on sensitivity ORDER, repairs integer mean to exactly
   --rank-mean 8. Mixed arm now has EXACTLY the same bpw as uniform r=8.
   The two arms are a clean rank-budget-matched ablation.
3. Path discipline: scripts live in /content/colab on Colab and
   /kaggle/working on Kaggle — every driver invocation uses the right
   prefix. vm_lead.sh uses PY=/content/colab.
4. Colab ssh: NEVER `colab ssh -s <name>` (creates a NEW CPU VM); always
   `ssh -o ProxyCommand='colab ssh --proxy-mode -s %h' SESSION`. Wrapper:
   `bash colab/vm_ssh.sh '<cmd>'` (env ISOQ_SESSION to override).
5. HF org is `Solstice-AI`, user `SolsticeAI` (token works on both dirs;
   Solace-1.0-Omni gated=auto, accessible, OpenAI-messages schema,
   parquet-backed so streaming works).

### Renames (do not regress)

- Paper title + method: **QuATernaryons / QuATernaryon** (user, 2026-09-29;
  was QuaTernaryions; was IsoQuant-Extreme). 7 occurrences in main.tex,
  bib header, vm_qtern.sh. PDF rebuilt.
- NOTE: user's HF demo repo is `SolsticeAI/QuATernaryions-demo` (ions) —
  name mismatch vs paper is known, repo rename is the user's call.

### Section P of findings.md — IsoQuant (Ji) full read

arXiv 2603.28430 is KV-stage-1-only, synthetic-only, random (not
calibrated) quaternion blocks. Our differentiation stands; see §P for the
positioning + numbers. The planned KV-cache quantized eval is our
differentiator — build QuantizedCache + long-context eval after weights
ladder lands.

### Old Colab notes (superseded but keep the ssh gotcha)

The rest of this file (§1-§9) is the 2026-09-24 record. Compute sections
there are stale; this §0 is authoritative.

---

---

## 0b. One-paragraph state (2026-09-24, STALE) (MORNING, superseded where §9 conflicts)

## 1. Mission & targets (user directives, verbatim intent)

- Author "IsoQuant-Extreme" paper: dual-sided sub-2-bit weight + dynamic KV
  quantization via 4D isoclinic Clifford rotors. LaTeX in `paper/main.tex`
  (build: `~/tectonic paper/main.tex`).
- User demands REAL numbers on REAL models. NOT synthetic-only.
- Target bit-width: **~2.1–2.2 bpw, MUST stay ternary** ({-1,0,+1} base).
- Quality bar: retain ~97–99% of fp16 (short context); >90% is hard cutoff.
- Long context: Qwen3.8-27B supports 262K — long-ctx robustness is a first-
  class claim. No published sub-2-bit results exist at 128K+ (verified):
  that gap is the paper's opening.
- Deployment target: ternary-capable llama.cpp exists on user's machine, but
  our format does NOT load there — PyTorch eval IS the deployment analog
  for now. GGUF export is future work.
- Compute: user's M4 Mac is OFF LIMITS for heavy jobs ("give my machine
  peace"). Use Colab T4 (15GB VRAM). parallel-cli for web research.
- Data stance: do NOT train on narrow/niche data only — mix broad + code/
  math/agentic (user's org has datasets). Wikitext is outdated for TRAINING;
  keep it only as the eval yardstick (every comparison paper reports it).

## 2. Files that matter

- `colab/isoquant_weights.py` (~880 lines) — the harness. Phases:
  `calib` (collect per-Linear activation Gram matrices H = XX^T — these ARE
  the imatrix), `quant` (PTQ), `alloc` (mixed-precision budget), `eval`
  (PPL). Key options: `--rotor {cal,hadamard,none}`, `--bits 1.62` (ternary2
  path), `--rank 8` / `--imatrix-r` (imatrix-mixed rank allocation),
  `--quant-sub` (output dir), `--raw` (baseline eval).
- `colab/isoquant_qat.py` — QAT polish. Trains ONLY A (per-group alpha),
  B (per-row beta), U,V (rank-r factors). T codes + rotor R frozen.
  NO STE needed: reconstruction is differentiable in these params.
  Loss: CE + 0.5·KL(student||fp16-teacher). Streaming data mix via
  `--mix "HuggingFaceFW/fineweb-edu:0.6,Solstice-AI/Solace-1.0-Omni:0.4"`.
  Drift guard: `--val-every 100` prints held-out wikitest KL.
- `colab/gen_demo.py` — generation demo (`--raw` or `--quant <subdir>`).
- `verify_isoquant_math.py` — verified math suite, 22/22 green, DO NOT
  modify without rerunning. The harness imports `rotor_butterfly`,
  `hadamard` from it via `_verified_suite()`.
- `colab/test_isoquant_weights.py` — 18/18 harness unit tests.
- `colab/vm_all.sh` — the full VM pipeline (currently running).
- `research/findings.md` — literature log. §K/§L = 2026 landscape, §M =
  QAT-polish pivot + long-ctx risk study. READ §§K–M before writing paper.
- `research/real_model_results.md` + `.json` — evidence log for numbers.
- `ckpt_smol/` (local M4) — SmolLM2 Hessians (96 .pt) + completed arms
  `quant_none` (PPL 104837.73 — real, see §4), `quant_hadamard` (13.25),
  `quant_mixed` (7.4084 @ 3.90 bpw uniform mixed), `quant_ternary` v1
  (broken, see §4), `bits_map.json`.## 3. VM access & operations (2026-09-24 late — WORKING RECIPE)

**CORRECTION to §9.4: compute is NOT blocked.** Colab T4 sessions do not
consume CU from the user's balance ("t4s dont use any credits theyre free"),
so `Current balance: 0.00 compute units` is a non-issue for T4. `colab new
-s <name> --gpu T4` creates a working T4 even at 0 CU.

**The GPU-in-ssh fix (this burned hours; do not lose it again).** The ssh
shell IS the GPU runtime container — it is NOT a separate CPU sidecar. But
`nvidia-smi` fails ("couldn't find libnvidia-ml.so") and
`torch.cuda.is_available()` returns **False** because the driver libs are
not on the default search path. The host driver libs live in
`/usr/lib64-nvidia/` (NOT `/usr/local/cuda-12.8/compat` — that one gives
torch error 803 "unsupported display driver / cuda driver combination").

```bash
export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH
# verify: python3 -c "import torch;print(torch.cuda.is_available())"  -> True
```
Put that export at the top of every VM script and every remote one-liner.
Host driver: 580.82.07. Container torch: 2.11.0+cu128, python 3.13.

**Working ssh command** (session name substitutable):
```bash
ssh -o User=root -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
    -o ProxyCommand="colab ssh --proxy-mode -s %h" isoq2
```
`~/.ssh/config` may hold a stale `colab` alias pointing at a dead session —
check `colab sessions` for the live name first. Bare `colab ssh --proxy-mode`
(no `-s`) auto-creates a *different* CPU VM — never use it.

**What does NOT work (don't retry):** `colab exec -s NAME -f script.py` —
dies with `AttributeError: 'WebSocketApp' object has no attribute
'send_text'` (local `websocket-client` 1.6.1 is too old for jkc 0.15.0's
wsclient; needs >=1.7). Upgrading it risks the ssh path, so the kernel
transport is abandoned in favour of ssh. `colab console -s NAME` reaches the
same container as ssh (same hostname) but is a tmux TTY — awkward to script.

**Launching long jobs:** background them so they survive the ssh channel;
the channel is held open by an un-disowned child, so expect the launching
ssh call to hang until you ctrl-c / the tool times out — the job still runs.
```bash
cd /content && setsid nohup bash vm_all.sh > /content/all.log 2>&1 < /dev/null &
```

**Push code:** tar+scp, then flatten into /content:
```bash
tar czf /tmp/push.tgz colab/isoquant_weights.py colab/isoquant_qat.py \
    colab/gen_demo.py verify_isoquant_math.py
scp -o User=root -o ProxyCommand="colab ssh --proxy-mode -s %h" /tmp/push.tgz isoq2:/content/
ssh ... isoq2 'cd /content && tar xzf push.tgz && mv colab/*.py .'
```

**Sessions are reclaimed without warning.** Harvest results to the local
repo the moment they appear; never let evidence exist only on /content.

### 3.2 QAT script had three crash bugs (never executed before 2026-09-24)

`isoquant_qat.py` was written and syntax-checked but never run; the first
real execution died immediately. All fixed — but note the lesson: **py_compile
proves nothing about a data path.**
1. `load_dataset(name, split=split, ...)` — `split` was never defined
   (NameError) in the streaming/mix branch.
2. `AutoModelForCausalLM` was used for the teacher but only `AutoTokenizer`
   was imported inside `main()` (NameError).
3. `refill()` reassigned `cum` without `nonlocal cum`, making it local to
   `refill`; the first `np.searchsorted(cum, ...)` raised UnboundLocalError.
Also discovered: `Solstice-AI/Solace-1.0-Omni` is a **gated** dataset
("You must be authenticated to access it"), so the QAT mix needs an HF token
on whatever machine runs it. `vm_lead.sh` falls back to fineweb-edu-only
when the mix fails, so the headline numbers can still land without it.

### 3.0 Colab T4 quota is EXHAUSTED today (2026-09-24 late)

Symptom fingerprint of a reclaimed runtime: ssh AND `colab console` AND
`colab exec` all report
`[colab] Session 'X' appears to be lost (404/401). Cleaning up.`
while `colab sessions` still lists the dead runtime as `[?]`. That `[?]`
entry is cosmetic — `colab stop -s X` says "not found".

Diagnosed: a **CPU** runtime still creates and stops fine, so this is NOT a
session-count problem; `colab new -s Y --gpu T4` is refused with
"Allocation refused (precondition failed) … too many active sessions, or a
temporary usage or capacity limit" and `--gpu L4` says the account has no
entitlement. Conclusion: the free T4 GPU hours for the account are spent
for now (today burned ~3 T4 sessions, two of which were reclaimed mid-job).
Wait for the quota to reset, or move the campaign to Modal (see §9.4).

**A reclaimed session loses ALL of /content.** Nothing on the VM is durable:
Hessians, quant shards, logs. Always pull artifacts to the local repo as they
appear (the Hessians are only ~3 min to recompute, but quant shards are
~70-90s of CPU each).

**Do not run a second heavy process alongside the main job.** A 3-step QAT
smoke test launched next to the running pipeline very likely OOM'd the
12 GB host RAM (student + teacher = two 1.7B bf16 models plus the numpy quant
process) and that is the best explanation for the second loss.

### 3.1 The CPU-vs-GPU bug (why every earlier "T4" run was slow and died)

The harness hardcoded `torch.device("mps" if torch.backends.mps.is_available()
else "cpu")`. On the Colab VM MPS does not exist, so **every calib/eval/QAT
run silently fell through to CPU** — the T4 sat at 0% util, 3 MiB used, while
a python3 process burned 95% CPU. That is the real reason a single calib seq
took 419s and why runs outlived their sessions.

Fixed by a single helper in `isoquant_weights.py`:
```python
def pick_device(requested="mps"):
    # honors an explicit request if available, else cuda > mps > cpu
```
`s isoquant_weights.py`: calib + `eval_ppl` now use it (`--device` gained a
`cuda` choice). `isoquant_qat.py` and `gen_demo.py` call `iw.pick_device`.
`vm_all.sh` passes `--device cuda` explicitly on every torch phase.

**Measured before/after:** `calib seq 0/89` went 419s -> **1s**.

**Always verify the device is really in use**, don't trust that it started:
```bash
LD_LIBRARY_PATH=/usr/lib64-nvidia nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
```
Expect non-zero util and ~3-4 GB used. Zero util + one pegged python3 = you
are on CPU again.

Model on VM: /root/.cache/huggingface/hub/models--HuggingFaceTB--SmolLM2-1.7B
/snapshots/effd688a12921b4cc83e3312b6feb579f70f9c71 (path also in
/content/MODEL_DIR.txt). Same snapshot as local M4.

Push code changes: `scp -q colab/isoquant_weights.py colab:/content/` etc.
The running bash script does NOT re-read files per phase — if you change
code mid-run, either restart the remaining phases manually or let it finish
and patch after.

## 4. Verified findings (do not re-litigate)

1. **Hessian = imatrix.** Our calib Hessians have ~10,000x diag dynamic
   range (outlier channels). rotor=none (no rotation) GPTQ shatters:
   PPL 104,837.73 vs hadamard 13.25 — that explosion is REAL (verified by
   three independent computations agreeing to 6 decimals; the earlier
   "eval bug" suspicion was wrong — my probe had a wrong normalizer,
   tr(WHW^T) instead of tr(H)).
2. **Error metric**: rel² = tr(E H E^T)/tr(H), H = rotated Hessian in the
   basis where E lives. Flat 2-bit GPTQ (hadamard): per-module rel² ~0.7–1.5
   → PPL 13.25. Uniform mixed-precision alloc (target 0.03, kappa=0.8348):
   avg 3.90 bpw, PPL 7.4084 (+2.59% vs 7.2212, model predicted 2.97% —
   alloc machinery VALIDATED end-to-end).
3. **Cross-term trap (critical!)**: after GPTQ feedback, correcting
   individual residual ENTRIES raises total error even when exact
   (2.52 → 3.61 measured with EXACT fp16 correction of top-1% entries).
   E H E^T is not separable; GPTQ feedback creates beneficial correlations.
   => NO sparse-entry residual channel in rotated basis. Rank-r correction
   is the only compatible form (subspace-level, preserves cross-terms).
4. **Rank-r curve** (layer0.q_proj, cal rotor, ternary base rel²=2.52):
   r=4 → 0.851, r=8 → 0.764, r=16 → 0.679, r=64 → 0.480 (fp16 factors).
   int4 factors at r=8: 0.994 (bpw 2.071). fp16 r=8: 0.764 (bpw 2.165).
5. **Honest bpw accounting** (the old 1.62-bpw claim was false — it omitted
   per-group scale bits and used a 40-bit sparse channel):
   ternary2 = entropy(~1.53) + 16/g (alpha per-32-group) + 16/out (beta
   per-row) + 16·r·(1/out+1/in) (U,V fp16). 2048², g=32, r=8 → **~2.17 bpw
   all-in**. This is the honest number; per-row beta (PT2-LLM's mu) saves
   ~0.5 bpw vs per-group beta.
6. **v1 ternary path is broken** (PPL would be garbage — demo produced
   gibberish): its residual targeted the feedback-mutated matrix
   (double-counts propagated error) AND used one global int8 scale
   (Hessian 10,000x range picks tiny residuals the coarse scale overshoots).
   v2 (`ternary_quantize_module2`) fixes target (pristine Wr − base) and
   drops sparse residual entirely for rank-r. KEEP `Wfb`/`Wr` separation —
   this bug bit twice; guard it forever.
7. **imatrix-r allocation** (`--imatrix-r`): per-module effective rank
   eff = exp(−Σ p·log p) of diag(H) distribution → rank table
   [(24,4),(8,8),(3,16),(1.2,32),(0,48)]: concentrated (salient) modules
   get MORE correction budget, diffuse get less; bpw ~constant. Verified:
   q_proj eff=14.65→r8, v_proj eff=126→r4.
8. **QAT polish is fully differentiable**: Wq = B + repeat(A)·T + U·V^T in
   rotated basis; T,R frozen ⇒ gradient flows to A,B,U,V with zero tricks.
   Motivation: pure PTQ plateaus (base ternary rel² 2.52 even with
   feedback; only structural corrections help). EfficientQAT (arXiv
   2407.11062) proves tiny-QAT recovers most degradation (<3% @ 2-bit,
   single GPU). Literature: §M of findings.md.
9. **Long-ctx quant risk** (arXiv 2505.20276): 4-bit loses up to 23%
   retrieval at 128K (grows with length); 8-bit ~lossless; mechanism
   hypothesis = RoPE error accumulation ⇒ RoPE-aware mixed precision
   (q/k_proj get more budget than v/o) + train quant params on mixed
   lengths. TODO on Qwen track.
10. **Calibration data choice matters** (arXiv 2311.09755): substantial
    downstream variation across calibration sets. Use on-distribution mix.
11. **fp16 storage of the Hessians was a latent crash, now fixed.**
    `collect_hessians` saved each H with `torch.save(Hm.to(torch.float16))`.
    With a ~10,000x diagonal dynamic range, fp16 rounding destroys the small
    eigenvalues and leaves a *numerically indefinite* matrix, so
    `cholesky` dies. Observed live: `linAlgError: Matrix is not positive
    definite` at `fit_rotation` -> `np.linalg.cholesky(H64)`. It is
    module-specific (layer0.k_proj has min eig 9.6e-5 and survives;
    the next module does not), which is why a run could look healthy for a
    few modules and then explode. Fixes, all three applied:
    (a) Hessians now saved **fp32** (96 x 2048^2 x 4B = 1.6 GB, fine);
    (b) `fit_rotation` symmetrizes H and falls back to an eigen-clipped
        symmetric square root when cholesky fails (identical on success);
    (c) new `_inv_cholesky()` helper does the same for the GPTQ feedback
        factor `U` (used by `gptq_quantize` and `ternary_quantize_module2`).
    Any Hessian written before this fix must be recomputed — do not reuse.

## 5. Running pipeline & expected outputs

`/content/vm_all.sh` (running, log /content/all.log):
- Phase 2 calib: 96 Hessians, `CALIB_EXIT=0`. WARNING: calib seq 0 took
  419s (warmup incl. wikitext download + model load); if seq 1+ don't
  drop to ~15–30s, kill and rerun calib with `--n-seqs 128`.
- Phase 3a: quant_ternary2 (uniform r=8) — expect per-module e≈0.7–2.5.
- Phase 3b: quant_ternary2m (imatrix-r mixed ranks).
- Phase 4: QAT 400 steps each on fineweb-edu stream (NOTE: vm_all.sh
  still calls isoquant_qat.py with default single-dataset mode — the mix
  flag was added AFTER launch; if you want the Solace mix, rerun QAT
  manually with --mix, see §6).
- Phase 5 evals: baseline PPL (expect ≈7.22), quant_ternary2, _m, _q, _mq,
  then generation demos. Harvest into research/real_model_results.md.

Success bar: quant_ternary2q/_mq PPL ≤ ~7.6 (≈5% deg) = QAT works and we
scale to Qwen3.8-27B. PPL > 9 = QAT insufficient, debug before scaling.

## 6. Immediate next actions (in order)

1. Poll: `printf 'grep -aE "EXIT=|PPL|ALL_DONE" /content/all.log | tail\nexit\n' | ssh colab`.
2. When ALL_DONE: pull `/content/all.log` locally (scp), extract 5 PPLs +
   demo outputs; append to research/real_model_results.md with honest bpw
   (recompute from shards: entropy+0.5+0.008+16r(2/2048)).
3. Rerun QAT with the data mix (code now supports it):
   `python3 isoquant_qat.py --model-dir $(cat /content/MODEL_DIR.txt) \
    --ckpt /content/ckpt --quant-sub quant_ternary2m \
    --out-sub quant_ternary2mq2 --steps 600 --lr 1e-3 --seq 1024 \
    --batch 1 --kl 0.5 \
    --mix "HuggingFaceFW/fineweb-edu:0.6,Solstice-AI/Solace-1.0-Omni:0.4" \
    --val-every 100`
   (HF auth: user is `SolsticeAI`, logged in on local; VM may need
   `huggingface-cli login` token for the org datasets if they are gated.)
4. If PPL improves: lock recipe, write Qwen3.8-27B pipeline (§7).
5. Paper: update Table 4 + abstract with REAL SmolLM2 numbers (baseline
   7.2212, flat-2bit 13.25, uniform-mixed 3.90bpw 7.4084, ternary arms);
   cite QTEA (2609.00224), PT2-LLM (2510.03267), CALDERA, EfficientQAT,
   2505.20276 long-ctx study. Rebuild with ~/tectonic.

## 7. Qwen3.8-27B track (staged, not started)

Model: user's own line — e.g. `Solstice-AI/Qwen3.8-27B-TURBO-...-GGUF-...`
(GGUF! need fp16/bf16 safetensors instead — check `unsloth` or base
`Qwen/Qwen3.8-27B` mirrors; base PPL 6.9505 published by Unsloth).
Memory plan on T4 (15GB):
- calib: use `--load-4bit` (bitsandbytes NF4, CUDA-only path exists) —
  ~15.5GB, tight; or L4 (24GB) if quota allows.
- quant: streams modules from safetensors — fine as-is. 6144-dim: H is
  6144² fp16 = 75MB × 192 modules ≈ 14GB disk — add fp16 H loading
  (already saved fp16; ensure quant loads fp16→float64 per module then
  frees). eigh per module on 6144² twice (effective_error + rank_residual)
  ≈ 30–60s each × 192 → 2–3h acceptable on T4.
- QAT: NF4-frozen backbone + trainable A/B/U/V only. Est. params for 27B:
  ~400M trainable in quant params? NO — compute: out·(n_g+1)+r·(out+in)
  per module ≈ 33k–100k × modules; total < 20M. Fine on T4 with offload.
- Long-ctx: eval at 32K windows (stride), needle probes if time permits;
  q/k modules get +1 rank tier (RoPE insurance).
Colab notebook deliverable for the user's own runs (they asked before).

## 8. User profile & collaboration notes

- User is sharp, impatient, hates wasted time: "sleep less", "search the
  web, there's been so much advancement", "use parallel cli". DEFAULT to
  parallel-cli for research (`parallel-cli search "objective" -q kw --json
  -o /tmp/f.json` then read the file; ALWAYS verify claims — user trusts
  verified numbers, catches hand-waving).
- User catches real errors (spot the MPS-vs-CUDA, the missing GPU, the
  imatrix insight, the QAT pivot, "wikitext is outdated"). When they
  redirect, PIVOT IMMEDIATELY and honestly — do not defend dead approaches.
- Their HF org: Solstice-AI (models: Qwen3.8-27B line incl. NVFP4/GGUF/
  MLX variants; datasets: Axiom-1.0, Complete-FABLE.5-traces-2M,
  Solace-1.0-Omni 322 chunks of agentic/code/math trace JSONL
  {"messages":[...]}, Solace-270K-Golden-131K-SFT).
- User collection "Project Solace" (4 items) mixes frontier distillations.
- Terse replies okay; no fluff; show real numbers; be honest about broken
  things (user respects "that was broken, here's the fix" >> fake progress).
- Hand off tasks cleanly (this file) when session gets long.
