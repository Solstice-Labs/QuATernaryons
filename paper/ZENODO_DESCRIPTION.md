QuATernaryons introduces dual-sided sub-2-bit weight and dynamic KV-cache quantization using 4D isoclinic Clifford rotors (quaternion left/right multiplication) composed into fast O(d log d) rotor-butterfly networks. Each rotor is mathematically orthogonal by construction, uses 8 floats instead of 16, and inverts trivially by quaternion conjugation.

What's New in Version 0.2.0:
1. SmolLM2-1.7B Real-Model Gates: First zero-external-data distribution-fidelity gates at 2.18 bpw across all 96 attention projections (70.2% top-1 agreement; 79.9% with zero-data polish at 0.134 nats/token KL divergence).
2. 27B Attention Benchmarks: 7.0% error reduction over identity baseline on trained 27B checkpoint projections.
3. Hardware Kernels: Complete kernel implementations provided for Apple Silicon (Metal Shading Language) and NVIDIA GPUs (Triton).
4. Dual-Sided KV Proofs: Machine-verified proofs for score invariance, post-RoPE consistency, and softmax stability.
5. Complete Verification: 22/22 numerical and algebraic checks verified to machine precision.
