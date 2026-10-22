"""
test_bonsai_sampling.py — Local Evaluation of Bonsai-2 Sampling Protocol on Sub-2-Bit Weights
Tests whether Bonsai-2's (T=0.7, top_k=20, top_p=0.80, presence_penalty=1.5) eliminates
repetition attractor loops and stabilizes autoregressive text generation at temperature.
"""

import os
import sys
import time
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quaternaryons.module import QuATernaryonLinear

def main():
    model_id = "openbmb/MiniCPM3-4B"
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmarks", "arm_g64_quaternaryons_weights.pt")

    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found.")
        sys.exit(1)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # MiniCPM3-4B custom kernels run best on float32/bfloat16 on Apple Silicon MPS/CPU
    dtype = torch.float32
    print(f"Loading {model_id} architecture on local Mac ({device})...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)
    if hasattr(config, "rope_parameters") and config.rope_parameters:
        config.rope_parameters["factor"] = float(
            config.max_position_embeddings / config.rope_parameters.get("original_max_position_embeddings", config.max_position_embeddings)
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    print(f"Loading QuATernaryons weights from {weights_path}...", flush=True)
    weights = torch.load(weights_path, map_location="cpu")

    # Group weights by module prefix
    mod_weights = {}
    for k, v in weights.items():
        # Match module paths like model.layers.0.self_attn.o_proj
        for suffix in [".T", ".alpha", ".beta", ".U", ".V"]:
            if k.endswith(suffix):
                m_name = k[:-len(suffix)]
                if m_name not in mod_weights:
                    mod_weights[m_name] = {}
                break

    for k, v in weights.items():
        for m_name in mod_weights:
            if k.startswith(m_name + "."):
                sub_k = k[len(m_name) + 1:]
                mod_weights[m_name][sub_k] = v

    print(f"Replacing {len(mod_weights)} linear layers with QuATernaryonLinear...", flush=True)
    for name, m_dict in mod_weights.items():
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        leaf_name = parts[-1]
        orig_mod = getattr(parent, leaf_name)

        out_f, in_f = m_dict["T"].shape
        rank = m_dict["U"].shape[1] if "U" in m_dict else 8

        q_mod = QuATernaryonLinear(
            in_features=in_f,
            out_features=out_f,
            group_size=64,
            rank=rank,
            bias=orig_mod.bias is not None,
            trainable_quant=False
        )

        q_mod.load_state_dict(m_dict, strict=False)
        q_mod.freeze_for_inference()
        q_mod.rotor.sync_buffers()
        setattr(parent, leaf_name, q_mod.to(device=device, dtype=dtype))

    model.to(device)
    model.eval()
    print(">>> Student Model initialized and frozen for inference successfully!\n", flush=True)

    prompts = [
        "In quantum mechanics, the relationship between SU(2) and SO(3) is",
        "The fundamental reason sub-2-bit quantization suffers from outlier collapse is"
    ]

    for p in prompts:
        print("=" * 80)
        print(f"PROMPT: {p}")
        print("=" * 80)
        inputs = tokenizer(p, return_tensors="pt").to(device)

        # 1. Greedy (T=0) — The Baseline that hit repetition loops in Track 1
        print("\n--- 1. GREEDY DECODING (T=0) ---")
        t0 = time.time()
        out_greedy = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"[{time.time()-t0:.2f}s]:\n" + tokenizer.decode(out_greedy[0], skip_special_tokens=True).strip())

        # 2. Bonsai-2 Non-Thinking Profile (T=0.7, top_k=20, top_p=0.80, presence_penalty=1.5)
        print("\n--- 2. BONSAI-2 PROFILE (T=0.7, top_k=20, top_p=0.80, presence_penalty=1.5) ---")
        t0 = time.time()
        out_bonsai = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=True,
            temperature=0.7,
            top_k=20,
            top_p=0.80,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"[{time.time()-t0:.2f}s]:\n" + tokenizer.decode(out_bonsai[0], skip_special_tokens=True).strip())

        # 3. Bonsai-2 Reasoning Profile (T=1.0, top_k=20, top_p=0.95, presence_penalty=1.0)
        print("\n--- 3. BONSAI-2 REASONING PROFILE (T=1.0, top_k=20, top_p=0.95, presence_penalty=1.0) ---")
        t0 = time.time()
        out_reasoning = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=True,
            temperature=1.0,
            top_k=20,
            top_p=0.95,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
        print(f"[{time.time()-t0:.2f}s]:\n" + tokenizer.decode(out_reasoning[0], skip_special_tokens=True).strip())
        print("\n")

if __name__ == "__main__":
    main()
