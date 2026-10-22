"""
quaternaryons.export_gguf — Standalone GGUF Model Exporter for QuATernaryons
Compiles quantized sub-2-bit weights into standard GGUF binary format with
high-density 5-trits-per-byte packing (1.6 bpw discrete grid) and continuous
FP16 metadata (alpha, beta, U, V, rotors) for local edge deployment.
"""

import os
import sys
import json
import argparse
from typing import Dict, Any, Optional

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer

try:
    from .pack import pack_trits_5
except ImportError:
    from quaternaryons.pack import pack_trits_5


def export_quaternaryons_to_gguf(
    weights_path: str,
    output_gguf_path: str,
    model_id: str = "openbmb/MiniCPM3-4B",
    hf_token: Optional[str] = None
):
    """
    Exports QuATernaryons weights into a standalone ~1.1 GB GGUF model.
    """
    try:
        import gguf
    except ImportError:
        raise ImportError("gguf package is required for GGUF export. Run `pip install gguf`.")
    print(f"[GGUF Exporter] Loading config & tokenizer for {model_id}...", flush=True)
    config = AutoConfig.from_pretrained(model_id, token=hf_token)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)

    print(f"[GGUF Exporter] Loading quantized weights from {weights_path}...", flush=True)
    weights = torch.load(weights_path, map_location="cpu")

    arch = "minicpm3"
    writer = gguf.GGUFWriter(output_gguf_path, arch)

    # 1. Add Architecture Metadata
    writer.add_name(f"MiniCPM3-4B-QuATernaryons-Sub2Bit")
    writer.add_description("MiniCPM3-4B quantized under QuATernaryons 4D Isoclinic Clifford Rotor Framework")
    writer.add_context_length(getattr(config, "max_position_embeddings", 4096))
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_layer_norm_rms_eps(getattr(config, "rms_norm_eps", 1e-6))

    # 2. Add Tokenizer Metadata
    tokens = []
    scores = []
    tok_types = []
    vocab_size = len(tokenizer)
    for i in range(vocab_size):
        text = tokenizer.decode([i])
        tokens.append(text.encode("utf-8", errors="replace"))
        scores.append(0.0)
        tok_types.append(1) # normal token

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(tok_types)
    if tokenizer.bos_token_id is not None:
        writer.add_bos_token_id(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        writer.add_eos_token_id(tokenizer.eos_token_id)

    # 3. Add Model Tensors
    print("[GGUF Exporter] Packing and writing model tensors...", flush=True)
    n_tensors = 0
    total_packed_bytes = 0

    for name, tensor in weights.items():
        if "Wq_eval" in name:
            continue

        if name.endswith(".T"):
            # Dense 5-trits-per-byte packing
            packed, orig_shape = pack_trits_5(tensor)
            # Store uint8 trits viewed as int8 (GGUF supported int8)
            arr = packed.numpy().view(np.int8)
            tensor_name = name.replace(".T", ".T_packed_5trit")
            writer.add_tensor(tensor_name, arr)
            total_packed_bytes += arr.nbytes
            n_tensors += 1
        else:
            # Continuous parameters (alpha, beta, U, V, rotors, norms) in FP16
            if tensor.dtype in (torch.bfloat16, torch.float32):
                arr = tensor.to(torch.float16).numpy()
            elif tensor.dtype == torch.int8:
                arr = tensor.numpy()
            else:
                arr = tensor.numpy()
            writer.add_tensor(name, arr)
            total_packed_bytes += arr.nbytes
            n_tensors += 1

    print(f"[GGUF Exporter] Writing binary header, metadata, and {n_tensors} tensors...", flush=True)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    out_size_mb = os.path.getsize(output_gguf_path) / 1e6
    print(f"\n======================================================================")
    print(f"  >>> GGUF EXPORT COMPLETE: {output_gguf_path}")
    print(f"      Total Tensors: {n_tensors}")
    print(f"      File Size:     {out_size_mb:.2f} MB ({out_size_mb / 1024:.2f} GB)")
    print(f"======================================================================\n", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export QuATernaryons weights to GGUF")
    parser.add_argument("--weights", type=str, required=True, help="Path to input .pt weights")
    parser.add_argument("--output", type=str, required=True, help="Path to output .gguf file")
    parser.add_argument("--model-id", type=str, default="openbmb/MiniCPM3-4B")
    args = parser.parse_args()

    export_quaternaryons_to_gguf(args.weights, args.output, args.model_id)
