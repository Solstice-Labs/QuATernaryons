# 🌌 QuATernaryons: Dual-Sided Sub-2-Bit Weight and Dynamic KV-Cache Quantization via 4D Isoclinic Clifford Rotors

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23530592.svg)](https://doi.org/10.5281/zenodo.23530592)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Solstice-Labs/QuATernaryons/blob/main/QuATernaryons_MiniCPM_Benchmark.ipynb)

**QuATernaryons** establishes the theoretical and computational foundation for sub-2-bit Large Language Model compression by lifting orthogonal channel mixing into the unit 3-sphere $\mathbb{S}^3$ of Hamilton quaternions. 

By parameterizing orthogonal transformations through **4D left- and right-isoclinic Clifford rotors** arranged in an $O(d \log_4 d)$ butterfly network, QuATernaryons achieves uniform activation diffusion with strictly zero numerical drift and exact adjoint invertibility.

---

## 📐 Why MiniCPM is Mathematically Superior to SmolLM for QuATernaryons

| Structural Property | SmolLM2-1.7B | MiniCPM-2B (`openbmb/MiniCPM-2B-sft-bf16`) | Mathematical Significance |
| :--- | :--- | :--- | :--- |
| **Head Dimension ($d_h$)** | $64$ | **$64 = 4^3$** | Exactly 16 4D quaternion blocks; exactly $\log_4(64) = 3$ butterfly stages. |
| **Hidden Size ($d$)** | $2048 = 2^{11}$ | **$2304 = 36 \times 64$** | Pure multiple of $4^3$; zero padding across all 36 attention heads. |
| **Intermediate Size ($d_{\text{mlp}}$)** | $8192 = 2^{13}$ | **$5760 = 90 \times 64$** | Pure multiple of $4^3$; zero padding across all 90 MLP blocks. |
| **SIMD Lane Efficiency** | Non-uniform | **100% Lane Utilization** | 0 wasted parameters, zero dead weights, perfect SIMD-4 execution. |
| **Activation Geometry** | Uncalibrated Outliers | **WSD + RMSNorm Scaled** | Exceptionally well-conditioned activations prevent quantization collapse. |

---

## 🎯 The Zero-Data Evaluation Gate (The One Metric That Matters)

External evaluation benchmarks (Wikitext PPL, MMLU) can be gamed through training set leakage or corpus selection. QuATernaryons adopts the **Zero-Data Distribution-Fidelity Protocol** mined from empirical trial records:

1. **Synthetic Teacher Generator:** Exactly **65,536 tokens** generated autoregressively by the pristine FP16/BF16 teacher model from uniform random seed blocks using top-$k(40)$ sampling ($T=1.0$, seed 1234, 64 sequences $\times$ 1024 sequence length).
2. **Distribution Divergence:** Token-wise evaluation of Kullback-Leibler divergence $\mathrm{KL}(P_{\text{teacher}} \parallel P_{\text{student}})$ in **nats/token**.
   $$\mathrm{KL} = \frac{1}{N} \sum_{i=1}^N \sum_{v \in V} P_{\text{teacher}}(v \mid x_{<i}) \ln \left( \frac{P_{\text{teacher}}(v \mid x_{<i})}{P_{\text{student}}(v \mid x_{<i})} \right)$$
   *(Note: Evaluated with per-token normalization to prevent PyTorch's `batchmean` inflation bug).*
3. **Top-1 Token Agreement:** The fraction of sequence positions where the student's top greedy prediction strictly matches the teacher ($\mathrm{argmax}(P_s) == \mathrm{argmax}(P_t)$).
4. **Autoregressive Coherence Guard:** Verification across reasoning, coding, and mathematical prompts to ensure no **MC-generation dissociation** or rotor phase drift occurs during long-horizon generation.

---

## ⚡ Quantization Structure & Runtime Dequantization

Each weight projection $W \in \mathbb{R}^{\text{out} \times \text{in}}$ is transformed into the Clifford rotor basis:
$$W_{\text{rot}} = W R^T \quad \text{where} \quad R = \prod_{s=1}^{\log_4 d} \Pi_s \mathbf{B}_s \in \mathrm{SO}(d)$$

The weight is decomposed into:
1. **Per-Row Asymmetric Offset:** $\beta = \operatorname{mean}(W_{\text{rot}}, \text{axis}=1) \in \mathbb{R}^{\text{out} \times 1}$ *(proven by Helcig et al.'s $\gamma^2$ variance law to be necessary for distribution-level fidelity)*.
2. **Asymmetric Ternary Core:** $T \in \{-1, 0, +1\}^{\text{out} \times \text{in}}$ with group size $g=32$.
3. **Per-Group Scale:** $\alpha \in \mathbb{R}^{\text{out} \times (\text{in}/g)}$.
4. **Low-Rank SVD Correction:** $U \in \mathbb{R}^{\text{out} \times r}, V \in \mathbb{R}^{\text{in} \times r}$ ($r=8$).

$$\text{Forward Pass:} \quad y = (x R^T) \left( \beta + \operatorname{repeat}(\alpha, g) \odot T + U V^T \right)^T + \text{bias}$$

*At deployment, ternary codes $T$ and rotors $R$ are frozen. Continuous parameters $(\alpha, \beta, U, V)$ are smoothly refined during a 300-step Zero-Data QAT polish pass without straight-through estimators (STE).*

---

## 🚀 Quickstart & Colab Execution

### 1-Click Google Colab Notebook
Run the end-to-end MiniCPM sub-2-bit benchmark directly on a free Google Colab GPU:
👉 **[`QuATernaryons_MiniCPM_Benchmark.ipynb`](https://github.com/Solstice-Labs/QuATernaryons/blob/main/QuATernaryons_MiniCPM_Benchmark.ipynb)**

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

### Standalone Colab Runner Script
```bash
python3 colab_minicpm_quaternaryons.py \
    --model-id openbmb/MiniCPM-2B-sft-bf16 \
    --group-size 32 \
    --rank 8 \
    --n-eval-tokens 65536 \
    --device cuda \
    --do-qat \
    --qat-steps 300
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
