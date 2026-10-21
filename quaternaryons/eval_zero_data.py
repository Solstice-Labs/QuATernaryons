"""
quaternaryons.eval_zero_data — The Zero-Data Synthetic Evaluation Gate
The definitive metric for distribution-level quantization fidelity mined from the project archives.
Evaluates:
  1. KL(P_teacher || P_student) in nats/token on 65,536 teacher-generated tokens.
  2. Top-1 Token Agreement percentage (OrcaSAQ2-class standard).
  3. Autoregressive Coherence Guard (prevents MC-generation dissociation).
"""

import time
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


def generate_teacher_pool(
    teacher_model,
    tokenizer,
    num_tokens: int = 65536,
    seq_len: int = 1024,
    device: str = "cuda",
    blk_size: int = 32,
    batch_size: int = 8,
    seed: int = 1234,
    top_k: int = 40
) -> torch.Tensor:
    """
    Generates synthetic tokens drawn strictly from the teacher's own distribution.
    Zero external corpus — reference == generator == evaluation domain.
    """
    vocab_size = teacher_model.config.vocab_size
    g = torch.Generator(device="cpu").manual_seed(seed)
    num_sequences = num_tokens // seq_len
    rows = []
    done = 0
    t0 = time.time()

    print(f"[eval_zero_data] Generating {num_tokens} synthetic tokens ({num_sequences} sequences)...", flush=True)

    while done < num_sequences:
        b = min(batch_size, num_sequences - done)
        # Uniform initial random seed block of tokens
        x = torch.randint(0, vocab_size, (b, blk_size), generator=g).to(device)

        # Autoregressively extend up to seq_len using teacher's top-k(40) distribution
        while x.size(1) < seq_len:
            curr_blk = min(blk_size, seq_len - x.size(1))
            with torch.no_grad():
                logits = teacher_model(x).logits[:, -1, :].float()  # (b, V)
            k = min(top_k, logits.size(-1))
            top_vals, _ = torch.topk(logits, k, dim=-1)
            # Truncate below top-k
            logits = torch.where(
                logits < top_vals[..., -1:],
                torch.full_like(logits, float("-inf")),
                logits
            )
            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=g)  # (b, 1)
            x = torch.cat([x, nxt], dim=1)

        rows.append(x.cpu())
        done += b
        print(f"  Generated {done * seq_len}/{num_tokens} tokens ({time.time()-t0:.1f}s)", flush=True)

    return torch.cat(rows, dim=0)  # (num_sequences, seq_len)


@torch.no_grad()
def evaluate_zero_data_distribution(
    teacher_model,
    student_model,
    pool_tokens: torch.Tensor,
    device: str = "cuda",
    micro_slice: int = 512
) -> Tuple[float, float]:
    """
    Measures KL(P_t || P_s) in nats/token and Top-1 token agreement.
    Avoids PyTorch's F.kl_div(batchmean) normalization pitfall.
    """
    tot_kl = 0.0
    tot_agree = 0
    tot_tokens = 0

    teacher_model.eval()
    student_model.eval()

    num_seqs = pool_tokens.size(0)
    for i in range(num_seqs):
        x = pool_tokens[i:i+1].to(device)  # (1, seq_len)
        with torch.no_grad():
            ht = teacher_model(x).logits.float()
            hs = student_model(x).logits.float()

        seq_len = ht.size(1)
        for s0 in range(0, seq_len - 1, micro_slice):
            s1 = min(s0 + micro_slice + 1, seq_len)
            lt = ht[:, s0:s1]  # (1, chunk, V)
            ls = hs[:, s0:s1]

            # Shift position by 1 for next-token prediction
            lt_pred = lt[:, :-1]
            ls_pred = ls[:, :-1]
            n_tok = lt_pred.size(1)

            # Log-softmax over vocabulary
            logp_t = torch.log_softmax(lt_pred, dim=-1)
            logp_s = torch.log_softmax(ls_pred, dim=-1)
            p_t = torch.softmax(lt_pred, dim=-1)

            # KL(t || s) = sum_v P_t(v) * (log P_t(v) - log P_s(v))
            kl_step = (p_t * (logp_t - logp_s)).sum(dim=-1).mean().item()
            agree_step = (lt_pred.argmax(dim=-1) == ls_pred.argmax(dim=-1)).sum().item()

            tot_kl += kl_step * n_tok
            tot_agree += agree_step
            tot_tokens += n_tok

    avg_kl = tot_kl / max(tot_tokens, 1)
    top1_acc = tot_agree / max(tot_tokens, 1)
    return avg_kl, top1_acc


@torch.no_grad()
def evaluate_autoregressive_coherence(
    model,
    tokenizer,
    prompts: Optional[List[str]] = None,
    max_new_tokens: int = 128,
    device: str = "cuda"
) -> Dict[str, str]:
    """
    Autoregressive generation test to check for MC-generation dissociation / rotor drift.
    """
    if prompts is None:
        prompts = [
            "Explain how quaternion algebra parameterizes 4D rotations:",
            "Write a Python function to compute the Fibonacci sequence efficiently:",
            "The fundamental reason sub-2-bit quantization suffers from outlier collapse is",
            "In quantum mechanics, the relationship between SU(2) and SO(3) is"
        ]

    results = {}
    model.eval()
    for p in prompts:
        inputs = tokenizer(p, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding to test stability
            pad_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        results[p] = gen_text
    return results
