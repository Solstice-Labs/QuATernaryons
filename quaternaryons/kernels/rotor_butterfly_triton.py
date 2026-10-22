"""
quaternaryons.kernels.rotor_butterfly_triton
Triton implementation of per-head 4D isoclinic Clifford rotor butterfly stage
fused with 2-bit affine dequantization (BLOCK=4 aligned).
"""

import triton
import triton.language as tl


@triton.jit
def rotor_butterfly_kv2bit(
    X_ptr,        # *i8   packed 2-bit K or V, row-major [T, d]
    S_ptr, Z_ptr, # *fp32 per-group scale / zero [d/G]
    Q_ptr,        # *fp32 unit quaternions [d//4, 4]
    P_ptr,        # *i32  stride permutation [d]
    Y_ptr,        # *fp32 output [T, d]
    T, D, G: tl.constexpr,
    BLOCK_T: tl.constexpr
):
    pid_t = tl.program_id(0)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)      # rows
    offs_d = tl.arange(0, D)                               # channels
    mask_t = offs_t < T

    # 1. Groupwise affine dequantization (levels {0,1,2,3} -> FP32)
    codes = tl.load(X_ptr + offs_t[:, None] * D + offs_d[None, :],
                    mask=mask_t[:, None], other=0)
    g = offs_d // G
    s = tl.load(S_ptr + g)
    z = tl.load(Z_ptr + g)
    x = s[None, :] * codes.to(tl.float32) - z[None, :]

    # 2. 4-wide rotor butterflies, log4(D) stages with register residency
    for t in tl.static_range(6):
        x = tl.reshape(x, (BLOCK_T, D // 4, 4))
        q0 = tl.load(Q_ptr + (tl.arange(0, D // 4) * 4 + 0))[:, None]
        q1 = tl.load(Q_ptr + (tl.arange(0, D // 4) * 4 + 1))[:, None]
        q2 = tl.load(Q_ptr + (tl.arange(0, D // 4) * 4 + 2))[:, None]
        q3 = tl.load(Q_ptr + (tl.arange(0, D // 4) * 4 + 3))[:, None]
        a, b, c, e = tl.split(x, 4)
        y0 = q0 * a - q1 * b - q2 * c - q3 * e
        y1 = q1 * a + q0 * b - q3 * c + q2 * e
        y2 = q2 * a + q3 * b + q0 * c - q1 * e
        y3 = q3 * a - q2 * b + q1 * c + q0 * e
        x = tl.reshape(tl.join(y0, y1, y2, y3), (BLOCK_T, D))
        # Stride permutation between butterfly stages
        p = tl.load(P_ptr + offs_d)
        x = tl.gather(x, p[None, :].broadcast_to(BLOCK_T, D), axis=1)

    tl.store(Y_ptr + offs_t[:, None] * D + offs_d[None, :], x, mask=mask_t[:, None])
