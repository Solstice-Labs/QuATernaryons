"""
quaternaryons.pack — High-Density 5-Trit-per-Byte Dense Packing
Packs ternary codes T in {-1, 0, 1} at 5 trits per 8-bit byte:
    3^5 = 243 <= 255.
Achieves an exact discrete bitrate of 8 / 5 = 1.60 bits per weight.
"""

from typing import Tuple
import torch


def pack_trits_5(T: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Packs ternary tensor T with values in {-1, 0, 1} into a dense uint8 1D array.
    Returns:
        packed: 1D uint8 tensor of length ceil(T.numel() / 5)
        shape: original shape tuple of T
    """
    orig_shape = tuple(T.shape)
    t_flat = (T.flatten() + 1).to(torch.uint8)  # map {-1, 0, 1} -> {0, 1, 2}
    numel = t_flat.numel()
    rem = numel % 5
    pad = (5 - rem) % 5
    if pad > 0:
        pad_tensor = torch.ones(pad, dtype=torch.uint8, device=T.device)
        t_flat = torch.cat([t_flat, pad_tensor], dim=0)

    c = t_flat.reshape(-1, 5)
    packed = (
        c[:, 0]
        + 3 * c[:, 1]
        + 9 * c[:, 2]
        + 27 * c[:, 3]
        + 81 * c[:, 4]
    ).contiguous()
    return packed, orig_shape


def unpack_trits_5(packed: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Unpacks a dense uint8 1D array back to the original ternary tensor in {-1, 0, 1}.
    """
    original_numel = 1
    for dim in shape:
        original_numel *= dim

    c0 = packed % 3
    r1 = torch.div(packed, 3, rounding_mode="floor")
    c1 = r1 % 3
    r2 = torch.div(r1, 3, rounding_mode="floor")
    c2 = r2 % 3
    r3 = torch.div(r2, 3, rounding_mode="floor")
    c3 = r3 % 3
    c4 = torch.div(r3, 3, rounding_mode="floor")

    unpacked = torch.stack([c0, c1, c2, c3, c4], dim=-1).flatten()[:original_numel]
    return (unpacked.to(torch.int8) - 1).reshape(shape)
