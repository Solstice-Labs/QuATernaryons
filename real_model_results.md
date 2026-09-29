# Real-model results: IsoQuant-Extreme pipeline on Bonsai-27B weights

Source: `Bonsai-27B-dspark-bf16.gguf` (llama.cpp GGUF, mmap-streamed).
Pipeline: identical to verify_isoquant_math.py — 6-stage rotor butterfly,
coarse-first strides, moment-equalizer fit + quantization-aware refinement;
2-bit group-32 symmetric quantizer. k/v rotated on the OUTPUT side
(per-head KV-store fuse), o on the INPUT side.

| layer | tensor | rotated dim | identity | Hadamard | rotor(rand) | rotor(calibrated) |
|---|---|---|---|---|---|---|
| 0 | attn_k | 512 | 0.5029 | 0.4541 | 0.4561 | 0.4389 |
| 0 | attn_v | 512 | 0.5576 | 0.4536 | 0.4560 | 0.4508 |
| 0 | attn_output | 5120 | 0.5244 | 0.4561 | 0.4611 | 0.4580 |
| 2 | attn_k | 512 | 0.4741 | 0.4561 | 0.4558 | 0.4478 |
| 2 | attn_v | 512 | 0.4919 | 0.4546 | 0.4557 | 0.4509 |
| 2 | attn_output | 5120 | 0.4657 | 0.4550 | 0.4573 | 0.4562 |
| 3 | attn_k | 512 | 0.4956 | 0.4541 | 0.4550 | 0.4496 |
| 3 | attn_v | 512 | 0.4618 | 0.4552 | 0.4546 | 0.4507 |
| 3 | attn_output | 5120 | 0.4608 | 0.4549 | 0.4568 | 0.4558 |
| 5 | attn_k | 512 | 0.5031 | 0.4544 | 0.4558 | 0.4499 |
| 5 | attn_v | 512 | 0.4400 | 0.4558 | 0.4555 | 0.4505 |
| 5 | attn_output | 5120 | 0.4411 | 0.4551 | 0.4592 | 0.4558 |

**Means (all):** identity 0.4849  hadamard 0.4549  rotor_rand 0.4566  rotor_calibrated 0.4512
**Means (k/v only):** identity 0.4909  hadamard 0.4547  rotor_rand 0.4556  rotor_calibrated 0.4486

Scope note: weight-side reconstruction on real trained matrices. These are
NOT model perplexities; KV-activation calibration requires runtime traces.
