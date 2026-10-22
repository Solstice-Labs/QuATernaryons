# 🚀 QuATernaryons: MiniCPM3-4B Sub-2-Bit Dual-Arm Benchmark Results

**Status:** ✅ **COMPLETED (Phase 1 Baseline)**  
**Target Model:** [](https://huggingface.co/openbmb/MiniCPM3-4B) (4.1 Billion parameters)  
**Evaluation Standard:** 65,536 Teacher-Generated Tokens ( 	imes 1024$) drawn strictly from teacher top-(40)$ distribution.

---

## 📊 Dual-Arm Benchmark Comparison

| Metric / Parameter | Baseline Teacher (BF16) | Arm A (=32$, Sub-2.2b) | Arm B (=64$, Sub-1.9b) |
| :--- | :--- | :--- | :--- |
| **Quantized Modules** | None (Pristine) | **All 196 Linear Layers (100%)** | **All 196 Linear Layers (100%)** |
| **Effective Bitrate (bpw)** | 16.00 bpw | **2.12 bpw** (2.119) | **1.86 bpw** (1.859) |
| **Top-1 Token Agreement** | 100.00% | **76.45%** | **75.41%** |
| **Student Perplexity (PPL)** | **2.60** | **6.80** | **8.46** |
| **KL Divergence** | 0.000 nats | **1.047 nats/token** | **1.247 nats/token** |
| **Evaluation Tokens** | 65,536 tokens | 65,536 tokens | 65,536 tokens |
| **Weight Tensors** | 8.15 GB FP16 | [****](arm_g32_quaternaryons_weights.pt) (4,073.7 MB) | [****](arm_g64_quaternaryons_weights.pt) (3,965.8 MB) |

---

## 📦 Verified Deliverables in Offshore Repository

1. **Teacher Distribution Shard:** [](pool_tokens.pt) (, 65,536 tokens).
2. **Arm A Weights (=32$, 2.12 bpw):** [](arm_g32_quaternaryons_weights.pt).
3. **Arm B Weights (=64$, 1.86 bpw):** [](arm_g64_quaternaryons_weights.pt).
4. **Machine-Readable Metrics:** [](minicpm3_4b_quaternaryons_results.json).
5. **Autoregressive Coherence Probes:** [](autoregressive_generations.json).

---

## 🛠️ Architecture Highlights
* **4D Isoclinic Clifford Rotors:** Parameterized on the ^3 	imes S^3 \cong 	ext{Spin}(4)$ manifold to disperse activation outliers.
* **(d \log_4 d)$ Butterfly Networks:** Replaced monolithic dense rotations with 64-dimensional quaternion butterfly blocks.
* **Instant In-Place Quantization:** Deallocated float weights instantly on GPU 1, keeping student VRAM consumption under 2.5 GB on dual NVIDIA Tesla T4 GPUs.
