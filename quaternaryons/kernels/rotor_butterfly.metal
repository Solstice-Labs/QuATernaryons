#include <metal_stdlib>
using namespace metal;

// -----------------------------------------------------------------------------
// QuATernaryons: Isoclinic Clifford Rotor Butterfly Kernel (v0.2.0)
// Evaluates left-isoclinic 4D quaternion rotor multiplication on SIMD-4 vectors.
// Forward stage: y = A(L) x
// Computational intensity: 12 FMA via 4 SIMD-4 multiply-adds.
// -----------------------------------------------------------------------------

kernel void rotor_butterfly_f32(
    device const float4* x_in      [[buffer(0)]],   // [nRows * blocks]
    device const float4* qL        [[buffer(1)]],   // [blocks] unit quaternions
    device       float4* x_out     [[buffer(2)]],   // [nRows * blocks]
    constant uint& nRows           [[buffer(3)]],
    constant uint& blocks          [[buffer(4)]],
    uint2 tid                      [[thread_position_in_grid]]) // (row, blockIdx)
{
    const uint r = tid.x;
    const uint b = tid.y;
    if (r >= nRows || b >= blocks) return;

    float4 q = qL[b];
    // Each thread processes a permuted 4-channel group
    float4 v = x_in[r * blocks + b];

    // Left-isoclinic butterfly: y = A(L) x (Definition 1)
    // Sign pattern strictly preserved from Hamilton quaternion algebra
    float4 l  = q;
    float4 vn = float4(-v.y,  v.x,  v.w, -v.z);
    float4 vq = float4(-v.z, -v.w,  v.x,  v.y);
    float4 vp = float4(-v.w,  v.z, -v.y,  v.x);

    float4 y = l.xxxx * v + l.yyyy * vn + l.zzzz * vq + l.wwww * vp;
    x_out[r * blocks + b] = y;
}

// -----------------------------------------------------------------------------
// Inverse Stage: exact adjoint via quaternion conjugation
// qL <- conj(qL) = [q.x, -q.y, -q.z, -q.w]
// Conjugate inversion is mathematically exact with 0 algebraic distortion.
// -----------------------------------------------------------------------------
kernel void rotor_butterfly_inv_f32(
    device const float4* x_in      [[buffer(0)]],
    device const float4* qL        [[buffer(1)]],
    device       float4* x_out     [[buffer(2)]],
    constant uint& nRows           [[buffer(3)]],
    constant uint& blocks          [[buffer(4)]],
    uint2 tid                      [[thread_position_in_grid]])
{
    const uint r = tid.x;
    const uint b = tid.y;
    if (r >= nRows || b >= blocks) return;

    // Quaternion conjugate: negate imaginary components
    float4 q = float4(qL[b].x, -qL[b].y, -qL[b].z, -qL[b].w);
    float4 v = x_in[r * blocks + b];

    float4 l  = q;
    float4 vn = float4(-v.y,  v.x,  v.w, -v.z);
    float4 vq = float4(-v.z, -v.w,  v.x,  v.y);
    float4 vp = float4(-v.w,  v.z, -v.y,  v.x);

    float4 y = l.xxxx * v + l.yyyy * vn + l.zzzz * vq + l.wwww * vp;
    x_out[r * blocks + b] = y;
}
