#!/usr/bin/env python3
"""Harvest the Colab run log + quant shard logs into the evidence file.

Usage:
    python3 research/harvest_vm_log.py /tmp/all.log [--ckpt /tmp/ckpt]

Parses (a) `all.log` from vm_all.sh for the phase sentinels, device line,
per-arm PPL and the QAT drift-guard series, and (b) each arm's
`_quant_log.json` for per-module Hessian-weighted error and honest bpw.
Writes research/real_model_results_smollm2.{md,json}.
"""
import argparse
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

PPL_RE = re.compile(r"PPL = ([0-9]+\.[0-9]+)")
SENT_RE = re.compile(r"^\+? ?(VERIFY_EXIT|CALIB_EXIT|QUANT_EXIT|QUANTM_EXIT|"
                     r"QAT_EXIT|QATM_EXIT)=([0-9]+)")
EVAL_RE = re.compile(r"--quant-sub ([A-Za-z0-9_]+)|--raw\b")
DRIFT_RE = re.compile(r"\[drift-guard\] held-out wiki KL ([0-9.]+)")
CUDA_ERR = ("CUDA out of memory", "Traceback (most recent call last)")
DEV_RE = re.compile(r"(calib|eval|qat) device: (\S+)")


def parse_log(path):
    """Walk the log once, attaching each PPL to the eval command before it."""
    out = {"sentinels": {}, "devices": {}, "arms": {}, "order": [],
           "drift": [], "errors": [], "has_phase5": False}
    arm = None
    with open(path, "r", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            m = SENT_RE.search(line)
            if m:
                out["sentinels"][m.group(1)] = int(m.group(2))
            m = DEV_RE.search(line)
            if m:
                out["devices"][m.group(1)] = m.group(2)
            if "isoquant_weights.py eval" in line or "gen_demo.py" in line:
                out["has_phase5"] = True
            if "eval --model-dir" in line or ("--quant-sub" in line
                                              and "eval" in line):
                if "--raw" in line:
                    arm = "baseline_fp32"
                else:
                    mm = re.search(r"--quant-sub ([A-Za-z0-9_]+)", line)
                    arm = mm.group(1) if mm else arm
                if arm and arm not in out["order"]:
                    out["order"].append(arm)
            m = PPL_RE.search(line)
            if m and arm:
                out["arms"][arm] = float(m.group(1))
            m = DRIFT_RE.search(line)
            if m:
                out["drift"].append(float(m.group(1)))
            for e in CUDA_ERR:
                if e in line and "CUDA out of memory" in line:
                    out["errors"].append(line.strip()[:200])
    return out


def arm_stats(ckpt_dir, arm):
    """Per-module Hessian-weighted error + honest bpw from a quant arm."""
    log = os.path.join(ckpt_dir, arm, "_quant_log.json")
    if not os.path.exists(log):
        return None
    rows = json.load(open(log))
    errs = [r["err_hess"] for r in rows if "err_hess" in r]
    ranks = [r.get("rank") for r in rows]
    bpw = []
    d = os.path.join(ckpt_dir, arm)
    for fn in os.listdir(d):
        if fn.endswith(".npz"):
            import numpy as np
            try:
                bpw.append(float(np.load(os.path.join(d, fn))["bpw"]))
            except Exception:
                pass
    return {
        "n_modules": len(rows),
        "err_hess_mean": round(st.fmean(errs), 4) if errs else None,
        "err_hess_median": round(st.median(errs), 4) if errs else None,
        "err_hess_max": round(max(errs), 4) if errs else None,
        "rank_min": min(ranks) if ranks else None,
        "rank_max": max(ranks) if ranks else None,
        "bpw_mean": round(st.fmean(bpw), 4) if bpw else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--ckpt", default="/tmp/ckpt")
    ap.add_argument("--baseline", type=float, default=7.2212,
                    help="known fp32 baseline PPL for degradation %")
    args = ap.parse_args()

    log = parse_log(args.log)
    arms = {}
    for name, ppl in log["arms"].items():
        if name == "baseline_fp32":
            continue
        arms[name] = {"ppl": ppl}
        s = arm_stats(args.ckpt, name)
        if s:
            arms[name].update(s)
        base = log["arms"].get("baseline_fp32", args.baseline)
        arms[name]["degradation_pct"] = round(100 * (ppl / base - 1), 2)

    base_ppl = log["arms"].get("baseline_fp32")
    payload = {
        "model": "HuggingFaceTB/SmolLM2-1.7B",
        "eval": "wikitext-2-raw test, 50k tokens, window 2048 stride 1024",
        "device": log["devices"],
        "sentinels": log["sentinels"],
        "baseline_fp32_ppl": base_ppl,
        "known_local_baseline_ppl": args.baseline,
        "arms": arms,
        "qat_drift_guard_wiki_kl": log["drift"],
        "oom_errors": log["errors"],
        "complete": "ALL_DONE" in open(args.log, errors="replace").read(),
    }
    jpath = os.path.join(HERE, "real_model_results_smollm2.json")
    json.dump(payload, open(jpath, "w"), indent=1)

    lines = ["# IsoQuant-Extreme — real-model results: SmolLM2-1.7B (Colab T4)",
             "", f"Model: {payload['model']}  ",
             f"Eval: {payload['eval']}  ",
             f"Devices: {log['devices']}  ",
             f"Phase exit codes: {log['sentinels']}  ",
             f"Run complete: {payload['complete']}", ""]
    if base_ppl:
        lines += [f"**Baseline fp32 PPL = {base_ppl:.4f}**", ""]
    if arms:
        lines += ["| arm | PPL | degradation | mean err_hess | bpw | modules |",
                  "|---|---|---|---|---|---|"]
        for name, a in arms.items():
            lines.append(
                f"| {name} | {a['ppl']:.4f} | "
                f"{a.get('degradation_pct', float('nan')):+.2f}% | "
                f"{a.get('err_hess_mean', '-')} | "
                f"{a.get('bpw_mean', '-')} | {a.get('n_modules', '-')} |")
        lines.append("")
    if log["drift"]:
        lines += ["QAT drift-guard (held-out wiki KL, every 100 steps): "
                  + ", ".join(f"{v:.4f}" for v in log["drift"]), ""]
    if log["errors"]:
        lines += ["## Errors captured", ""] + \
                 [f"- {e}" for e in log["errors"]] + [""]
    lines += ["Quality bar: >=90% retained precision = degradation <= 11.1% "
              "(~PPL <= 8.02). Target ~97% = <= 3% (~7.44).", ""]
    mpath = os.path.join(HERE, "real_model_results_smollm2.md")
    open(mpath, "w").write("\n".join(lines))
    print(f"wrote {jpath}\nwrote {mpath}")
    print(json.dumps(payload, indent=1)[:2000])


if __name__ == "__main__":
    sys.exit(main())
