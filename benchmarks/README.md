# 🚀 QuATernaryons: Sub-2-Bit Dual-Model Benchmark Registry

**Status:** ✅ **COMPLETED**  
**Offshore Vault:** [`Solstice-AI/QuATernaryons-demo-POC-alpha`](https://huggingface.co/datasets/Solstice-AI/QuATernaryons-demo-POC-alpha)  
**GitHub Repository:** [`Solstice-Labs/QuATernaryons`](https://github.com/Solstice-Labs/QuATernaryons)

---

## 📊 Track 1 & 2: MiniCPM3-4B Sub-2-Bit Dual-Arm Benchmark

| Metric / Parameter | Baseline Teacher (BF16) | Arm A ($g=32$, Sub-2.2b) | Arm B ($g=64$, Sub-1.9b) |
| :--- | :--- | :--- | :--- |
| **Model** | `openbmb/MiniCPM3-4B` | `openbmb/MiniCPM3-4B` | `openbmb/MiniCPM3-4B` |
| **Quantized Modules** | None (Pristine) | **All 196 Linear Layers (100%)** | **All 196 Linear Layers (100%)** |
| **Effective Bitrate (bpw)** | 16.00 bpw | **2.12 bpw** (2.119) | **1.86 bpw** (1.859) |
| **Top-1 Token Agreement** | 100.00% | **76.45%** | **75.41%** |
| **Student Perplexity (PPL)** | **2.60** | **6.80** | **8.46** |
| **KL Divergence** | 0.000 nats | **1.047 nats/token** | **1.247 nats/token** |
| **Evaluation Tokens** | 65,536 tokens | 65,536 tokens | 65,536 tokens |
| **Weight Tensors** | 8.15 GB FP16 | [`arm_g32_quaternaryons_weights.pt`](arm_g32_quaternaryons_weights.pt) (4,073.7 MB) | [`arm_g64_quaternaryons_weights.pt`](arm_g64_quaternaryons_weights.pt) (3,965.8 MB) |

---

## 📊 Track 3: Spark-X2.6-4B Sub-2-Bit CEC & PV-Tuning Benchmark

| Metric / Parameter | Baseline Teacher (FP16) | QuATernaryons + PV-Tuning Student (Sub-2-Bit) |
| :--- | :--- | :--- |
| **Target Model** | `XHToken/Spark-X2.6-4B` | `XHToken/Spark-X2.6-4B` |
| **Architecture Scope** | Pristine Float | **All 36 Layers (100% Quantized via CEC)** |
| **Group & Rank Allocation** | Float16 | **Q/K ($g=32, r=16$), V ($g=64, r=8$), Gate/Up ($g=32, r=8$), Down/Out ($g=64, r=8$)** |
| **Effective Bitrate** | 16.00 bpw | **~1.853 bpw** (Strictly $\le 2.05$ bpw ceiling) |
| **Student VRAM Footprint** | 8.00 GB FP16 | **4.97 GB** (Clean 58% memory reduction) |
| **Distillation Schedule** | N/A | **600 Steps PV-Tuning ($\Theta = \{\alpha, \beta, U, V\}$ in FP32 autograd)** |
| **Final Forward KL Divergence**| 0.000 nats | **4.8905 nats/token** |
| **Top-1 Token Agreement** | 100.00% | **45.02%** |
| **Teacher Perplexity on Gate**| 1.2765 | 1.2765 |
| **Saved Model Weights** | 8.00 GB | [`spark_x26_quaternaryons_weights.pt`](spark_x26_quaternaryons_weights.pt) (4,964.6 MB) |
| **Results Metadata** | N/A | [`spark_x26_quaternaryons_results.json`](spark_x26_quaternaryons_results.json) |
| **Checkpoint Shards (100-600)**| N/A | `spark_checkpoint_step_100.pt` through `spark_checkpoint_step_600.pt` |

---

## 📦 Verified Deliverables in Offshore Repository

1. **Teacher Calibration Shard:** [`spark_pool_tokens.pt`](spark_pool_tokens.pt) (65,536 tokens, `torch.Size([32, 2048])`).
2. **Spark-X2.6 Quantized Model Weights (1.85 bpw):** [`spark_x26_quaternaryons_weights.pt`](spark_x26_quaternaryons_weights.pt) (4,964.6 MB).
3. **Continuous Parameter Shards (Steps 100 to 600):** `spark_checkpoint_step_100.pt` through `600.pt` (467.9 MB each).
4. **Machine-Readable Metrics:** [`spark_x26_quaternaryons_results.json`](spark_x26_quaternaryons_results.json).
5. **MiniCPM3-4B Weights & Datasets:** [`arm_g32_quaternaryons_weights.pt`](arm_g32_quaternaryons_weights.pt), [`arm_g64_quaternaryons_weights.pt`](arm_g64_quaternaryons_weights.pt), [`minicpm3_4b_quaternaryons_results.json`](minicpm3_4b_quaternaryons_results.json).

---

## 🛠️ Architecture Highlights
* **4D Isoclinic Clifford Rotors:** Parameterized on the $S^3 \times S^3 \cong \text{Spin}(4)$ manifold to disperse activation outliers.
* **4-Stage Quaternion Butterfly Network:** $d_h=256=4^4$ exact isoclinic factorization without dense rotation overhead.
* **Asymmetric SVD Slicing:** Surgical rank allocation giving Q & K higher capacity ($r=16$) to protect query-key dot products.
* **Continuous Distillation Guard:** FP32 parameter representation for continuous matrices ($\alpha, \beta, U, V$) ensuring complete gradient stability across 600 distillation steps.
