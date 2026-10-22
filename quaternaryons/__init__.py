"""
QuATernaryons — Dual-Sided Sub-2-Bit Weight and Dynamic KV-Cache Quantization via 4D Isoclinic Clifford Rotors
"""

from .rotor import (
    qmul_torch,
    qconj_torch,
    left_matrix_torch,
    right_matrix_torch,
    cayley_matrix_torch,
    rotor_procrustes_fit_torch
)
from .butterfly import (
    butterfly_permutation_torch,
    QuaternionButterflyBlock64,
    QuATernaryonRotorNetwork
)
from .quant import (
    ternary_entropy,
    quantize_ternary_asymmetric,
    fit_low_rank_residual,
    compute_bpw
)
from .module import (
    QuATernaryonLinear
)
from .eval_zero_data import (
    generate_teacher_pool,
    evaluate_zero_data_distribution,
    evaluate_autoregressive_coherence
)
from .pack import (
    pack_trits_5,
    unpack_trits_5
)
from .export_gguf import (
    export_quaternaryons_to_gguf
)

__version__ = "0.2.0"
__all__ = [
    "qmul_torch",
    "qconj_torch",
    "left_matrix_torch",
    "right_matrix_torch",
    "cayley_matrix_torch",
    "rotor_procrustes_fit_torch",
    "butterfly_permutation_torch",
    "QuaternionButterflyBlock64",
    "QuATernaryonRotorNetwork",
    "ternary_entropy",
    "quantize_ternary_asymmetric",
    "fit_low_rank_residual",
    "compute_bpw",
    "QuATernaryonLinear",
    "generate_teacher_pool",
    "evaluate_zero_data_distribution",
    "evaluate_autoregressive_coherence",
    "pack_trits_5",
    "unpack_trits_5",
    "export_quaternaryons_to_gguf"
]
