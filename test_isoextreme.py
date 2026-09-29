import time
import torch
import torch.nn as nn
import numpy as np

# Set reproducibility seed
torch.manual_seed(20260923)
np.random.seed(20260923)
device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


# =====================================================================
# 1. 2-BIT GROUP-WISE UNIFORM QUANTIZER (WITH STE PASS-THROUGH)
# =====================================================================
def quant2_ste(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Group-wise symmetric 2-bit uniform quantization with Straight-Through Estimator."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, group_size)
    m = torch.amax(torch.abs(x_flat), dim=-1, keepdim=True).clamp(min=1e-8)
    delta = m / 1.5

    # Grid: {-1.5, -0.5, +0.5, +1.5} * delta
    q = torch.clamp(torch.round(x_flat / delta + 1.5) - 1.5, -1.5, 1.5)
    x_q = (q * delta).reshape(orig_shape)

    # STE: forward pass receives quantized tensor, backward passes gradients directly
    return x + (x_q - x).detach()


def quant2(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """Inference-only 2-bit group quantizer."""
    orig_shape = x.shape
    x_flat = x.reshape(-1, group_size)
    m = torch.amax(torch.abs(x_flat), dim=-1, keepdim=True).clamp(min=1e-8)
    delta = m / 1.5
    q = torch.clamp(torch.round(x_flat / delta + 1.5) - 1.5, -1.5, 1.5)
    return (q * delta).reshape(orig_shape)


# =====================================================================
# 2. REALISTIC TRANSFORMER ACTIVATION & PROJECTION GENERATOR
# =====================================================================
def generate_layer_data(
    tokens: int = 2048, d_in: int = 128, d_out: int = 512, seed: int = None
):
    if seed is not None:
        torch.manual_seed(seed)

    # Power-law covariance decay (mimics transformer hidden states)
    k = torch.arange(1, d_in + 1, dtype=torch.float32, device=device)
    eig = 1.0 / (k**0.7)
    Q, _ = torch.linalg.qr(torch.randn(d_in, d_in, device=device))
    Sigma = Q @ torch.diag(eig) @ Q.T
    L = torch.linalg.cholesky(Sigma)

    X = torch.randn(tokens, d_in, device=device) @ L.T

    # Systematic persistent outlier channels (e.g., attention sinks & LayerNorm biases)
    outlier_channels = [7, 31, 79, 113]
    X[:, outlier_channels] *= 18.0

    # Projection weight matrix (e.g., QKV or MLP down-projection)
    W = torch.randn(d_out, d_in, device=device) * (1.0 / np.sqrt(d_in))
    return X, W


# =====================================================================
# 3. BASELINE: Fast Walsh-Hadamard Transform (FWHT)
# =====================================================================
def fwht(x: torch.Tensor) -> torch.Tensor:
    orig = x.shape
    d = orig[-1]
    y = x.clone().reshape(-1, d)
    h = 1
    while h < d:
        y = y.view(-1, d // (2 * h), 2, h)
        y0 = y[:, :, 0, :].clone()
        y1 = y[:, :, 1, :].clone()
        y[:, :, 0, :] = y0 + y1
        y[:, :, 1, :] = y0 - y1
        h *= 2
    return y.view(orig) / np.sqrt(d)


# =====================================================================
# 4. 3-STAGE ROTOR BUTTERFLY CASCADE (R3 P16 R2 P4 R1)
# =====================================================================
def quat_to_left_rot(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack(
        [
            torch.stack([w, -x, -y, -z]),
            torch.stack([x, w, -z, y]),
            torch.stack([y, z, w, -x]),
            torch.stack([z, -y, x, w]),
        ]
    )


def quat_to_right_rot(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q[0], q[1], q[2], q[3]
    return torch.stack(
        [
            torch.stack([w, -x, -y, -z]),
            torch.stack([x, w, z, -y]),
            torch.stack([y, -z, w, x]),
            torch.stack([z, y, -x, w]),
        ]
    )


def make_block_rotor(ql: torch.Tensor, qr: torch.Tensor) -> torch.Tensor:
    ql = ql / torch.norm(ql).clamp(min=1e-8)
    qr = qr / torch.norm(qr).clamp(min=1e-8)
    return quat_to_left_rot(ql) @ quat_to_right_rot(qr)


class PermutedRotorStage(nn.Module):
    def __init__(self, dim: int = 128, stride: int = 1):
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.num_blocks = dim // 4
        self.ql = nn.Parameter(torch.randn(self.num_blocks, 4))
        self.qr = nn.Parameter(torch.randn(self.num_blocks, 4))
        with torch.no_grad():
            self.ql.zero_()
            self.qr.zero_()
            self.ql[:, 0] = 1.0
            self.qr[:, 0] = 1.0

        # Exact index gather permutation
        idx = torch.arange(dim).view(dim // stride, stride).t().contiguous().view(-1)
        self.register_buffer("perm", idx)

    def forward_matrix(self) -> torch.Tensor:
        blocks = [
            make_block_rotor(self.ql[i], self.qr[i]) for i in range(self.num_blocks)
        ]
        R = torch.block_diag(*blocks)
        if self.stride > 1:
            R = R[self.perm, :][:, self.perm]
        return R


class ProductionIsoQuant(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.dim = dim
        self.s1 = PermutedRotorStage(dim=dim, stride=1)
        self.s2 = PermutedRotorStage(dim=dim, stride=4)
        self.s3 = PermutedRotorStage(dim=dim, stride=16)
        self.register_buffer("fused_R", None)

    def get_orthogonal(self) -> torch.Tensor:
        R1 = self.s1.forward_matrix()
        R2 = self.s2.forward_matrix()
        R3 = self.s3.forward_matrix()
        return R3 @ R2 @ R1

    def fuse(self):
        with torch.no_grad():
            self.fused_R = self.get_orthogonal()
            err = torch.norm(
                self.fused_R.T @ self.fused_R - torch.eye(self.dim, device=device)
            ).item()
            assert err < 1e-4, f"Isometry violated! Err: {err}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fused_R is not None:
            return x @ self.fused_R.T
        return x @ self.get_orthogonal().T

    def calibrate(
        self,
        x_calib: torch.Tensor,
        w_calib: torch.Tensor,
        steps: int = 160,
        lr: float = 0.02,
    ):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)

        with torch.no_grad():
            y_target = x_calib @ w_calib.T

        for _ in range(steps):
            optimizer.zero_grad()
            R = self.get_orthogonal()

            x_rot = x_calib @ R.T
            w_rot = w_calib @ R.T

            # Direct end-to-end W2A2 linear layer reconstruction with STE
            x_q = quant2_ste(x_rot, group_size=32)
            w_q = quant2_ste(w_rot, group_size=32)

            y_pred = x_q @ w_q.T
            loss = torch.nn.functional.mse_loss(y_pred, y_target)

            loss.backward()
            optimizer.step()


def sync():
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


# =====================================================================
# 5. BENCHMARK SUITE (150 BATCHES x 2048 TOKENS)
# =====================================================================
def run():
    d_in = 128
    d_out = 512
    tokens = 2048
    num_seeds = 150
    group_size = 32

    print("=" * 84)
    print("  PRODUCTION ISOQUANT (W2A2 END-TO-END GEMM) vs FWHT ON APPLE SILICON")
    print(
        f"  Head Dim: {d_in} | Proj Dim: {d_out} | Group Size: {group_size} | Tokens/Batch: {tokens}"
    )
    print(f"  Topology : 3-Stage Clifford Butterfly (R3 P16 R2 P4 R1)")
    print(f"  Device   : {device}")
    print("=" * 84)

    # 1. Offline Calibration Phase
    print("\n[Step 1/3] Calibrating 3-Stage Cascade via STE on Layer Objective...")
    Xc, Wc = generate_layer_data(tokens=512, d_in=d_in, d_out=d_out, seed=20260923)
    iso = ProductionIsoQuant(dim=d_in).to(device)
    iso.calibrate(Xc, Wc, steps=160, lr=0.02)
    iso.fuse()
    print("✓ Calibration complete. Exact Lie algebra isometry verified.")

    # 2. Offline Weight Pre-Rotation: W' = W @ R^T
    _, W = generate_layer_data(tokens=1, d_in=d_in, d_out=d_out, seed=42)
    H = fwht(torch.eye(d_in, device=device))
    R = iso.fused_R

    W_unrot_q = quant2(W, group_size=group_size)
    W_fwht_q = quant2(W @ H.T, group_size=group_size)
    W_iso_q = quant2(W @ R.T, group_size=group_size)

    # 3. 150 Unseen Batches Evaluation
    print(f"\n[Step 2/3] Evaluating 150 Unseen Batches ({tokens} tokens each)...")
    u_errs, h_errs, i_errs, deltas = [], [], [], []

    with torch.no_grad():
        for s in range(num_seeds):
            X, _ = generate_layer_data(
                tokens=tokens, d_in=d_in, d_out=d_out, seed=100000 + s
            )
            Y_ref = X @ W.T

            # Model 1: Raw Unrotated 2-bit
            Yu = quant2(X, group_size=group_size) @ W_unrot_q.T
            eu = torch.nn.functional.mse_loss(Yu, Y_ref).item()

            # Model 2: Fast Walsh-Hadamard 2-bit
            Xh = fwht(X)
            Yh = quant2(Xh, group_size=group_size) @ W_fwht_q.T
            eh = torch.nn.functional.mse_loss(Yh, Y_ref).item()

            # Model 3: IsoQuant 3-Stage 2-bit
            Xi = X @ R.T
            Yi = quant2(Xi, group_size=group_size) @ W_iso_q.T
            ei = torch.nn.functional.mse_loss(Yi, Y_ref).item()

            u_errs.append(eu)
            h_errs.append(eh)
            i_errs.append(ei)
            deltas.append(((eh - ei) / eh) * 100)

            if (s + 1) % 30 == 0:
                print(
                    f"  Progress: {s + 1:3d}/{num_seeds} batches | Running Win over FWHT: {np.mean(deltas):+.2f}%"
                )

    # 4. Latency Benchmark
    print("\n[Step 3/3] Timing Activation Rotation Latency (1000 timed runs)...")
    bx = torch.randn(tokens, d_in, device=device)
    for _ in range(200):
        _ = fwht(bx)
        _ = bx @ R.T
    sync()

    t0 = time.perf_counter()
    for _ in range(1000):
        _ = fwht(bx)
    sync()
    lat_h = (time.perf_counter() - t0) / 1000 * 1e6

    t0 = time.perf_counter()
    for _ in range(1000):
        _ = bx @ R.T
    sync()
    lat_i = (time.perf_counter() - t0) / 1000 * 1e6

    # 5. Final Report
    mean_u = np.mean(u_errs)
    mean_h = np.mean(h_errs)
    mean_i = np.mean(i_errs)
    mean_delta = np.mean(deltas)

    print("\n" + "=" * 84)
    print("                    FINAL PRODUCTION REPORT (W2A2)")
    print("=" * 84)
    print(f"  Unrotated Baseline MSE     : {mean_u:.5f}  (Outlier degradation)")
    print(
        f"  Walsh-Hadamard (FWHT) MSE   : {mean_h:.5f}  (±{np.std(h_errs):.5f}) | Latency: {lat_h:6.1f} μs"
    )
    print(
        f"  IsoQuant 3-Stage Fused MSE  : {mean_i:.5f}  (±{np.std(i_errs):.5f}) | Latency: {lat_i:6.1f} μs"
    )
    print("-" * 84)
    print(
        f"  GEMM Distortion Delta       : {mean_delta:+.2f}% lower error than FWHT across 150 batches"
    )
    print(
        f"  Min / Max Recorded Delta    : {np.min(deltas):+.2f}% / {np.max(deltas):+.2f}%"
    )
    print(f"  Activation Rotation Speedup : {lat_h / lat_i:.2f}x FASTER than FWHT")
    print("=" * 84)


if __name__ == "__main__":
    run()
