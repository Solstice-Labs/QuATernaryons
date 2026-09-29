# Real-model results: IsoQuant-Extreme pipeline on Bonsai-27B weights

Source: `Bonsai-27B-dspark-bf16.gguf` (llama.cpp GGUF, mmap-streamed).
Pipeline: identical to verify_isoquant_math.py — 6-stage rotor butterfly,
coarse-first strides, moment-equalizer fit + quantization-aware refinement;
2-bit group-32 symmetric quantizer. k/v rotated on the OUTPUT side
(per-head KV-store fuse), o on the INPUT side.

| layer | tensor | rotated dim | identity | Hadamard | rotor(rand) | rotor(calibrated) |
|---|---|---|---|---|---|---|
| 0 | attn_output | 5120 | 0.5244 | 0.4561 | 0.4605 | 0.4589 |

**Means (all):** identity 0.5244  hadamard 0.4561  rotor_rand 0.4605  rotor_calibrated 0.4589
**Means (k/v only):** identity nan  hadamard nan  rotor_rand nan  rotor_calibrated nan

Scope note: weight-side reconstruction on real trained matrices. These are
NOT model perplexities; KV-activation calibration requires runtime traces.
