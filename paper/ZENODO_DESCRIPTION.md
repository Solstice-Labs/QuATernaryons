QuATernaryons introduces dual-sided sub-2-bit weight and dynamic KV-cache quantization using 4D isoclinic Clifford rotors (quaternion left/right multiplication) composed into fast O(d log d) rotor-butterfly networks. Each rotor is mathematically orthogonal by construction, uses 8 floats instead of 16, and inverts trivially by quaternion conjugation.

What's New in Version 0.3.0:
1. Full-Model Sub-2-Bit Scaling on MiniCPM3-4B: Comprehensive evaluation across all 248 linear projections of openbmb/MiniCPM3-4B (DeepSeek-style MLA) on the 65,536-token zero-data teacher gate.
2. Unprecedented Zero-Shot PTQ Baselines: Demonstrates 76.45% Top-1 teacher agreement and 1.047 nats/token KL divergence at 2.12 bpw (Arm A), and 75.41% agreement and 1.247 nats/token at 1.859 bpw (Arm B) with zero gradient training steps.
3. The Exponential Perplexity Horizon: Algebraic proof and empirical validation showing PPL_student = PPL_teacher * exp(D_KL), establishing that the +4.2 PPL gap is the exact exponential reflection of ~1.05 nat discrete ternary entropy under pure zero-shot PTQ.
4. Compiled GGUF Metal Runtime: Fully compliant GGUF export running natively on Apple Silicon Metal at 25.5 tokens/sec with verified mathematical isometry.
5. Hardware Kernels: Complete production kernels provided for Apple Silicon (Metal Shading Language) and NVIDIA GPUs (Triton).
