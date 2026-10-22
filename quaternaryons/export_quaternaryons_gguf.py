"""
quaternaryons.export_quaternaryons_gguf — Compliant GGUF Exporter for QuATernaryons
Integrates the 248 sub-2-bit QuATernaryon layers into the standard MiniCPM3-4B GGUF
architecture using the native PrismML / llama.cpp conversion pipeline.
"""

import os
import sys
import time
import collections
from pathlib import Path
import torch

# Ensure prism-src and project root are in sys.path
PRISM_SRC = "/Users/shreyangondaliya/src/prism-src"
sys.path.insert(0, PRISM_SRC)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from conversion.base import ModelBase, gguf, logger
from conversion.minicpm import MiniCPM3Model
from quaternaryons.module import QuATernaryonLinear


class QuATernaryonMiniCPM3Model(MiniCPM3Model):
    """
    Subclasses MiniCPM3Model to seamlessly inject the reconstructed 4D Clifford
    rotor sub-2-bit weights into all 248 attention and MLP linear projections.
    """
    model_arch = gguf.MODEL_ARCH.MINICPM3

    def __init__(self, *args, quaternaryons_weights_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.quat_weights_path = quaternaryons_weights_path
        self.quat_modules = {}
        if self.quat_weights_path and os.path.exists(self.quat_weights_path):
            self._load_quaternaryons_weights()

    def _load_quaternaryons_weights(self):
        logger.info(f"[QuATernaryons] Loading quantized weights from {self.quat_weights_path}...")
        w = torch.load(self.quat_weights_path, map_location="cpu")
        mod_weights = collections.defaultdict(dict)
        for k, v in w.items():
            for tag in [".T", ".alpha", ".beta", ".U", ".V", ".rotor"]:
                idx = k.find(tag)
                if idx != -1:
                    m_name = k[:idx]
                    sub_k = k[idx + 1:]
                    mod_weights[m_name][sub_k] = v
                    break

        logger.info(f"[QuATernaryons] Grouped {len(w)} tensors into {len(mod_weights)} QuATernaryonLinear modules.")
        self.quat_modules = mod_weights

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None):
        # Check if this linear layer was quantized under QuATernaryons
        base_name = name[:-7] if name.endswith(".weight") else name
        if base_name in self.quat_modules:
            m_dict = self.quat_modules[base_name]
            out_f, in_f = m_dict["T"].shape
            rank = m_dict["U"].shape[1] if "U" in m_dict else 8

            q_mod = QuATernaryonLinear(
                in_features=in_f,
                out_features=out_f,
                group_size=64,
                rank=rank,
                bias=False,
                trainable_quant=False
            )
            q_mod.load_state_dict(m_dict, strict=False)
            q_mod.freeze_for_inference()
            q_mod.rotor.sync_buffers()

            # Fast einsum block multiplication with 64-dim Clifford rotor stack
            Wq = q_mod.Wq_eval
            R_stack = q_mod.rotor.R_stack.to(Wq.dtype)
            num_heads_64 = in_f // 64
            Wq_reshaped = Wq.reshape(out_f, num_heads_64, 64)
            with torch.no_grad():
                Weff = torch.einsum('ohd, hde -> ohe', Wq_reshaped, R_stack).reshape(out_f, in_f).detach().to(torch.float32)

            logger.info(f"[QuATernaryons] Injected reconstructed sub-2-bit weight for {base_name} (shape {Weff.shape})")
            data_torch = Weff

        yield from super().modify_tensors(data_torch, name, bid)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Export QuATernaryons MiniCPM3-4B to compliant GGUF")
    parser.add_argument("--weights", type=str, required=True, help="Path to arm_g64_quaternaryons_weights.pt")
    parser.add_argument("--snapshot", type=str, required=True, help="Path to HF model snapshot")
    parser.add_argument("--output", type=str, required=True, help="Path to output .gguf file")
    parser.add_argument("--outtype", type=str, default="bf16", choices=["bf16", "f16", "q8_0"])
    args = parser.parse_args()

    ftype_map = {
        "bf16": gguf.LlamaFileType.MOSTLY_BF16,
        "f16": gguf.LlamaFileType.MOSTLY_F16,
        "q8_0": gguf.LlamaFileType.MOSTLY_Q8_0,
    }

    dir_model = Path(args.snapshot)
    fname_out = Path(args.output)
    output_type = ftype_map[args.outtype]

    model_instance = QuATernaryonMiniCPM3Model(
        dir_model=dir_model,
        ftype=output_type,
        fname_out=fname_out,
        quaternaryons_weights_path=args.weights,
        is_big_endian=False,
        use_temp_file=False,
        eager=False,
    )

    logger.info("Exporting QuATernaryons model to GGUF...")
    t0 = time.time()
    with torch.inference_mode():
        model_instance.write()
    logger.info(f"Done in {time.time()-t0:.2f}s! Successfully exported to {fname_out}")


if __name__ == "__main__":
    main()
