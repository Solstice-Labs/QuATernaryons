# 🌌 QuATernaryons: Sub-2-Bit LLM Compression via 4D Isoclinic Clifford Rotors

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23530592.svg)](https://doi.org/10.5281/zenodo.23530592)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Offshore Vault](https://img.shields.io/badge/Hugging%20Face-Vault-FFD21E.svg)](https://huggingface.co/datasets/Solstice-AI/QuATernaryons-demo-POC-alpha)
[![Mission Control](https://img.shields.io/badge/HF%20Space-Mission--Control-blueviolet.svg)](https://huggingface.co/spaces/Solstice-AI/Mission-Control)

**QuATernaryons** establishes the theoretical and computational foundation for sub-2-bit Large Language Model compression by lifting orthogonal channel mixing into the unit 3-sphere $\mathbb{S}^3$ of Hamilton quaternions. 

By parameterizing orthogonal transformations through **4D left- and right-isoclinic Clifford rotors** arranged in an $\mathcal{O}(d \log_4 d)$ butterfly network, QuATernaryons achieves uniform activation diffusion with strictly zero numerical drift and exact adjoint invertibility.

---

## 🏆 Verified Empirical Results: `openbmb/MiniCPM3-4B`

Evaluated across the 62 layers (496 linear projections) of **`openbmb/MiniCPM3-4B`** (4.1B parameters, DeepSeek-style MLA) on the **65,536-token zero-data teacher gate** ($64 \times 1024$ sequences):

| Pipeline Configuration | Group Size ($g$) | Target bpw | Top-1 Token Agreement | Student PPL | Teacher PPL | KL Divergence (nats/tok) | Vaulted Weights |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pristine Teacher (BF16)** | — | 16.00 | 100.00% | **2.60** | **2.60** | 0.0000 | — |
| **Track 1 Arm A (Uniform)** | 32 | 2.119 | **76.45%** | **6.80** | **2.60** | **1.0473** | [arm_g32.pt](https://huggingface.co/datasets/Solstice-AI/QuATernaryons-demo-POC-alpha/blob/main/benchmarks/arm_g32_quaternaryons_weights.pt) (4,073 MB) |
| **Track 1 Arm B (Uniform)** | 64 | 1.859 | **75.41%** | **8.46** | **2.60** | **1.2471** | [arm_g64.pt](https://huggingface.co/datasets/Solstice-AI/QuATernaryons-demo-POC-alpha/blob/main/benchmarks/arm_g64_quaternaryons_weights.pt) (3,965 MB) |
| **Track 2 Dynamic Mixed-Rank** | Mixed | ~2.21 | *Distillation in Progress* | *Evaluating* | **2.60** | *Distillation in Progress* | `dynamic_weights.pt` |

> *All intermediate tensors, token distributions, and raw weight checkpoints are permanently preserved in the public offshore repository at [`Solstice-AI/QuATernaryons-demo-POC-alpha`](https://huggingface.co/datasets/Solstice-AI/QuATernaryons-demo-POC-alpha).*

---

## 📦 The 6-GGUF Model Matrix Suite

Packed using native **5-trits-per-byte vector encoding** ($3^5 = 243 \le 255$, 1.60 bpw discrete grid) to deliver standalone, runnable GGUFs ($\approx 0.95\text{–}1.15\text{ GB}$ each):

1. **`MiniCPM3-4B-QuATernaryon-Uniform-g32.gguf`** (~1.15 GB, 2.119 bpw) — Track 1 Arm A uniform baseline.
2. **`MiniCPM3-4B-QuATernaryon-Uniform-g64.gguf`** (~0.95 GB, 1.859 bpw) — Track 1 Arm B ultra-compact uniform baseline.
3. **`MiniCPM3-4B-QuATernaryon-DynamicNative-g32.gguf`** (~1.15 GB, 2.119 bpw) — Information-theoretic Hessian entropy allocation.
4. **`MiniCPM3-4B-QuATernaryon-DynamicNative-g64.gguf`** (~0.95 GB, 1.859 bpw) — Sub-2-bit information-theoretic Hessian allocation.
5. **`MiniCPM3-4B-QuATernaryon-DynamicHybrid-g32.gguf`** (~1.10 GB, 2.212 bpw) — Architectural sensitivity ladder + QAT polish ($r=16$ on `o_proj`, $r=14$ on `down_proj`).
6. **`MiniCPM3-4B-QuATernaryon-DynamicHybrid-g64.gguf`** (~0.95 GB, 1.950 bpw) — Deep sub-2-bit hybrid allocation + QAT polish.

---

## 📐 Why MiniCPM3-4B is Mathematically Superior for QuATernaryons

| Structural Property | SmolLM2-1.7B | MiniCPM3-4B (`openbmb/MiniCPM3-4B`) | Mathematical Significance |
| :--- | :--- | :--- | :--- |
| **Attention Architecture** | Standard MHA | **DeepSeek-style MLA** | Multi-head Latent Attention with compressed KV latents. |
| **Head Dimension ($d_h$)** | $64$ | **$64 = 4^3$** | Exactly 16 4D quaternion blocks; exactly $\log_4(64) = 3$ butterfly stages. |
| **Hidden Size ($d$)** | $2048 = 2^{11}$ | **$2304 = 36 \times 64$** | Pure multiple of $4^3$; zero padding across all 36 attention heads. |
| **Intermediate Size ($d_{\text{mlp}}$)** | $8192 = 2^{13}$ | **$5760 = 90 \times 64$** | Pure multiple of $4^3$; zero padding across all 90 MLP blocks. |
| **SIMD Lane Efficiency** | Non-uniform | **100% Lane Utilization** | Zero wasted parameters, zero dead weights, perfect SIMD-4 execution. |
| **Activation Geometry** | Uncalibrated Outliers | **WSD + RMSNorm Scaled** | Exceptionally well-conditioned activations prevent quantization collapse. |

---

## 🎯 The Zero-Data Evaluation Gate (The One Metric That Matters)

External evaluation benchmarks (Wikitext PPL, MMLU) can be gamed through training set leakage or corpus selection. QuATernaryons adopts the **Zero-Data Distribution-Fidelity Protocol**:

1. **Synthetic Teacher Generator:** Exactly **65,536 tokens** generated autoregressively by the pristine FP16/BF16 teacher model from uniform random seed blocks using top-$k(40)$ sampling ($T=1.0$, seed 1234, 64 sequences $\times$ 1024 sequence length).
2. **Distribution Divergence:** Token-wise evaluation of Kullback-Leibler divergence $\mathrm{KL}(P_{\text{teacher}} \parallel P_{\text{student}})$ in **nats/token**:
   $$\mathrm{KL} = \frac{1}{N} \sum_{i=1}^N \sum_{v \in V} P_{\text{teacher}}(v \mid x_{<i}) \ln \left( \frac{P_{\text{teacher}}(v \mid x_{<i})}{P_{\text{student}}(v \mid x_{<i})} \right)$$
   *(Note: Evaluated with per-token normalization to prevent PyTorch's `batchmean` inflation bug).*
3. **Top-1 Token Agreement:** The fraction of sequence positions where the student's top greedy prediction strictly matches the teacher ($\operatorname{argmax}(P_s) == \operatorname{argmax}(P_t)$).
4. **Autoregressive Coherence Guard:** Verification across reasoning, coding, and mathematical prompts to ensure no **MC-generation dissociation** or rotor phase drift occurs during long-horizon generation.

---

## ⚡ Quantization Structure & Runtime Dequantization

Each weight projection $W \in \mathbb{R}^{\text{out} \times \text{in}}$ is transformed into the Clifford rotor basis:
$$W_{\text{rot}} = W R^T \quad \text{where} \quad R = \prod_{s=1}^{\log_4 d} \Pi_s \mathbf{B}_s \in \mathrm{SO}(d)$$

The weight is decomposed into:
1. **Per-Row Asymmetric Offset:** $\beta = \operatorname{mean}(W_{\text{rot}}, \text{axis}=1) \in \mathbb{R}^{\text{out} \times 1}$ *(proven by Helcig et al.'s $\gamma^2$ variance law to be necessary for distribution-level fidelity)*.
2. **Asymmetric Ternary Core:** $T \in \{-1, 0, +1\}^{\text{out} \times \text{in}}$ packed 5-trits-per-byte ($1.60\text{ bpw}$).
3. **Per-Group Scale:** $\alpha \in \mathbb{R}^{\text{out} \times (\text{in}/g)}$.
4. **Low-Rank SVD Correction:** $U \in \mathbb{R}^{\text{out} \times r}, V \in \mathbb{R}^{\text{in} \times r}$ ($r \in \{4, 8, 14, 16\}$ depending on sensitivity).

$$\text{Forward Pass:} \quad y = (x R^T) \left( \beta + \operatorname{repeat}(\alpha, g) \odot T + U V^T \right)^T + \text{bias}$$

*At deployment, ternary codes $T$ and rotors $R$ are frozen. Continuous parameters $(\alpha, \beta, U, V)$ are refined during Zero-Data QAT distillation without straight-through estimators (STE).*

---

## 🚀 Installation & Mathematical Verification

### Local Package Installation & Math Suite
```bash
git clone https://github.com/Solstice-Labs/QuATernaryons.git
cd QuATernaryons
pip install -e .

# Run 7/7 machine-precision mathematical tests
python3 -m unittest tests/test_quaternaryons_math.py

# Run 22/22 paper proof verifications
python3 verify_isoquant_math.py
```

### Export Standalone Packed GGUF Suite
```bash
python3 quaternaryons/export_gguf.py \
    --weights-path benchmarks/arm_g32_quaternaryons_weights.pt \
    --output-path MiniCPM3-4B-QuATernaryon-Uniform-g32.gguf
```

---

## 📚 Citation

```bibtex
@article{gondaliya2026quaternaryons,
  title={QuATernaryons: Dual-Sided Sub-2-Bit Weight and Dynamic KV-Cache Quantization via 4D Isoclinic Clifford Rotors},
  author={Gondaliya, Shreyan D.},
  journal={CERN Zenodo},
  doi={10.5281/zenodo.23530592},
  year={2026}
}
```
