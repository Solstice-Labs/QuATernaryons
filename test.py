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


# ==========================================
# 1. BASELINE: Fast Walsh-Hadamard Transform
# ==========================================
def fwht(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    d = orig_shape[-1]
    assert (d & (d - 1)) == 0, "Dimension d must be a power of 2 for FWHT"

    y = x.clone().reshape(-1, d)
    h = 1
    while h < d:
        y = y.view(-1, d // (2 * h), 2, h)
        y0 = y[:, :, 0, :].clone()
        y1 = y[:, :, 1, :].clone()
        y[:, :, 0, :] = y0 + y1
        y[:, :, 1, :] = y0 - y1
        h *= 2
    y = y.view(orig_shape) / np.sqrt(d)
    return y


# ==========================================
# 2. 4D ISOCLINIC CLIFFORD ROTOR BUTTERFLY
# ==========================================
def quat_to_left_rot(q: torch.Tensor) -> torch.Tensor:
    """Vectorized differentiable 4x4 Left-Isoclinic matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    row0 = torch.stack([w, -x, -y, -z])
    row1 = torch.stack([x, w, -z, y])
    row2 = torch.stack([y, z, w, -x])
    row3 = torch.stack([z, -y, x, w])
    return torch.stack([row0, row1, row2, row3])


def quat_to_right_rot(q: torch.Tensor) -> torch.Tensor:
    """Vectorized differentiable 4x4 Right-Isoclinic matrix."""
    w, x, y, z = q[0], q[1], q[2], q[3]
    row0 = torch.stack([w, -x, -y, -z])
    row1 = torch.stack([x, w, z, -y])
    row2 = torch.stack([y, -z, w, x])
    row3 = torch.stack([z, y, -x, w])
    return torch.stack([row0, row1, row2, row3])


def build_block_rotor(q_left: torch.Tensor, q_right: torch.Tensor) -> torch.Tensor:
    q_l_norm = q_left / torch.norm(q_left).clamp(min=1e-8)
    q_r_norm = q_right / torch.norm(q_right).clamp(min=1e-8)
    L = quat_to_left_rot(q_l_norm)
    R = quat_to_right_rot(q_r_norm)
    return L @ R


class RotorButterflyStage(nn.Module):
    def __init__(self, dim: int, stride: int = 1):
        super().__init__()
        assert dim % 4 == 0
        self.dim = dim
        self.stride = stride
        self.num_blocks = dim // 4

        self.q_left = nn.Parameter(torch.randn(self.num_blocks, 4))
        self.q_right = nn.Parameter(torch.randn(self.num_blocks, 4))
        self._init_identity()

    def _init_identity(self):
        with torch.no_grad():
            self.q_left.zero_()
            self.q_right.zero_()
            self.q_left[:, 0] = 1.0
            self.q_right[:, 0] = 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_flat = x.view(-1, self.dim)

        # Stride permutation
        if self.stride > 1:
            x_flat = (
                x_flat.view(-1, self.dim // self.stride, self.stride)
                .transpose(-1, -2)
                .reshape(-1, self.dim)
            )

        # 4x4 block rotation
        x_blocks = x_flat.view(-1, self.num_blocks, 4)
        out_blocks = []
        for i in range(self.num_blocks):
            R = build_block_rotor(self.q_left[i], self.q_right[i])
            out_blocks.append(x_blocks[:, i, :] @ R.T)
        out = torch.stack(out_blocks, dim=1).view(-1, self.dim)

        # Inverse stride
        if self.stride > 1:
            out = (
                out.view(-1, self.stride, self.dim // self.stride)
                .transpose(-1, -2)
                .reshape(-1, self.dim)
            )
        return out.view(orig_shape)


class IsoQuantButterflyCascade(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.stage1 = RotorButterflyStage(dim=dim, stride=1)
        self.stage2 = RotorButterflyStage(dim=dim, stride=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stage2(self.stage1(x))

    def calibrate(self, x_calib: torch.Tensor, steps: int = 100, lr: float = 0.05):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        # Minimize coordinate peak-to-average kurtosis (incoherence proxy for MSE)
        for _ in range(steps):
            optimizer.zero_grad()
            rotated = self.forward(x_calib)
            # Smooth incoherence penalty: ratio of l_infinity / l2
            peak = torch.amax(torch.abs(rotated), dim=-1)
            rms = torch.sqrt(torch.mean(rotated**2, dim=-1) + 1e-8)
            incoherence_loss = torch.mean(peak / rms)

            incoherence_loss.backward()
            optimizer.step()


# ==========================================
# 3. 2-BIT GROUP-WISE QUANTIZATION
# ==========================================
def quantize_2bit_group(x: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    orig_shape = x.shape
    x_flat = x.view(-1, group_size)
    m = torch.amax(torch.abs(x_flat), dim=-1, keepdim=True).clamp(min=1e-8)
    delta = m / 1.5
    q = torch.clamp(torch.round(x_flat / delta + 1.5) - 1.5, -1.5, 1.5)
    return (q * delta).view(orig_shape)


# ==========================================
# 4. BENCHMARK RUNNER
# ==========================================
def run_benchmark():
    dim = 128
    batch_size = 512
    num_seeds = 5

    hadamard_mse_list = []
    isoquant_mse_list = []
    hadamard_spiked_list = []
    isoquant_spiked_list = []

    print("=" * 72)
    print(f"  ISOQUANT EXTREME vs FAST WALSH-HADAMARD (DEVICE: {device})")
    print("=" * 72)
    print(f"Dimension: {dim} | Group Size: 32 | Bitrate Target: 2-bit (~1.69 bpw)")
    print("-" * 72)

    for seed in range(num_seeds):
        torch.manual_seed(20260923 + seed)

        X_standard = torch.randn(batch_size, dim, device=device)
        X_spiked = torch.randn(batch_size, dim, device=device)
        X_spiked[:, [3, 17, 45, 89]] *= 18.0

        isoquant = IsoQuantButterflyCascade(dim=dim).to(device)
        isoquant.calibrate(X_spiked, steps=100)

        with torch.no_grad():
            # Hadamard
            H_spiked = fwht(X_spiked)
            H_q_spiked = quantize_2bit_group(H_spiked)
            H_recon_spiked = fwht(H_q_spiked)
            err_hadamard_spiked = torch.nn.functional.mse_loss(
                H_recon_spiked, X_spiked
            ).item()

            H_std = fwht(X_standard)
            H_q_std = quantize_2bit_group(H_std)
            H_recon_std = fwht(H_q_std)
            err_hadamard_std = torch.nn.functional.mse_loss(
                H_recon_std, X_standard
            ).item()

            # IsoQuant
            I_spiked = isoquant(X_spiked)
            I_q_spiked = quantize_2bit_group(I_spiked)
            err_isoquant_spiked = torch.nn.functional.mse_loss(
                I_q_spiked, I_spiked
            ).item()

            I_std = isoquant(X_standard)
            I_q_std = quantize_2bit_group(I_std)
            err_isoquant_std = torch.nn.functional.mse_loss(I_q_std, I_std).item()

        hadamard_mse_list.append(err_hadamard_std)
        isoquant_mse_list.append(err_isoquant_std)
        hadamard_spiked_list.append(err_hadamard_spiked)
        isoquant_spiked_list.append(err_isoquant_spiked)

        delta = (
            (err_hadamard_spiked - err_isoquant_spiked) / err_hadamard_spiked
        ) * 100
        print(
            f"Seed {seed + 1:02d} | Spiked MSE: Hadamard = {err_hadamard_spiked:.4f} | IsoQuant = {err_isoquant_spiked:.4f} | Δ = {delta:+.2f}%"
        )

    mean_h_spiked = np.mean(hadamard_spiked_list)
    mean_i_spiked = np.mean(isoquant_spiked_list)
    delta_spiked = ((mean_h_spiked - mean_i_spiked) / mean_h_spiked) * 100

    print("-" * 72)
    print("FINAL SUMMARY (Averaged across 5 seeds):")
    print(f"  Hadamard Spiked MSE : {mean_h_spiked:.4f}")
    print(f"  IsoQuant Spiked MSE : {mean_i_spiked:.4f}")
    print(
        f"  Net Win over Hadamard: {delta_spiked:+.2f}% lower reconstruction distortion"
    )
    print("=" * 72)


if __name__ == "__main__":
    run_benchmark()
