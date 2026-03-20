"""
End-to-end evaluation script for AWQ + RDQuant INT4/INT8 mixed-precision pipeline.

Usage:
    python -m rdquant.int4_eval

Steps:
    1. Load Qwen3-4B
    2. Compute AWQ scales (2 calibration samples)
    3. Quantize with RDQuant INT4/INT8 at 5.3 bpw
    4. Evaluate WikiText-2 PPL
    5. Compare with reference baselines
"""

from __future__ import annotations

import gc
import time

import torch


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "/root/autodl-tmp/Qwen3-4B"
    wikitext_path = "/root/data/datasets/wikitext/test.parquet"

    print("=" * 70)
    print("AWQ + RDQuant INT4/INT8 Mixed-Precision Pipeline Evaluation")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[1/5] Loading Qwen3-4B...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"      Loaded in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 2. Compute AWQ scales
    # ------------------------------------------------------------------
    print("\n[2/5] Computing AWQ scales (2 calibration samples)...")
    t0 = time.time()

    calib_texts = [
        "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols.",
        "In mathematics, a function is a relation between a set of inputs and a set of permissible outputs. Functions have the property that each input is related to exactly one output. The notation f(x) is commonly used to denote a function.",
    ]

    from rdquant.awq_scale import compute_awq_scales

    # Standard ignore patterns for Qwen models
    ignore_patterns = ["*lm_head*", "*embed*", "*norm*"]

    awq_scales = compute_awq_scales(
        model=model,
        tokenizer=tokenizer,
        calib_texts=calib_texts,
        max_samples=2,
        seq_length=512,
        ignore=ignore_patterns,
        device="cuda",
    )
    print(f"      Computed scales for {len(awq_scales)} layers in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 3. Quantize with RDQuant INT4/INT8 at 5.3 bpw
    # ------------------------------------------------------------------
    print("\n[3/5] Quantizing with RDQuant INT4/INT8 at 5.3 bpw...")
    t0 = time.time()

    from rdquant.int4_quant import quantize_model_int4

    # Move model to CPU for quantization to avoid OOM
    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    qmodel = quantize_model_int4(
        model=model,
        budget_avg_bits=5.3,
        formats=["INT4", "INT8"],
        ignore=ignore_patterns,
        awq_scales=awq_scales,
        int4_group_size=128,
    )
    print(f"      Quantized in {time.time() - t0:.1f}s")
    print("\n      Per-layer allocation summary:")
    qmodel.print_summary()

    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 4. Evaluate WikiText-2 PPL
    # ------------------------------------------------------------------
    print("\n[4/5] Evaluating WikiText-2 perplexity...")
    t0 = time.time()

    from rdquant.eval import eval_perplexity

    ppl = eval_perplexity(
        model=qmodel,
        tokenizer=tokenizer,
        dataset="wikitext",
        dataset_path=wikitext_path,
        seq_length=2048,
    )
    print(f"      WikiText-2 PPL: {ppl:.2f}  (computed in {time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Comparison table
    # ------------------------------------------------------------------
    print("\n[5/5] Comparison with baselines:")
    print("-" * 50)
    print(f"  {'Method':<35} {'PPL':>8}")
    print("-" * 50)
    print(f"  {'BF16 baseline':<35} {'12.90':>8}")
    print(f"  {'NVFP4/FP8 RDQuant data-free':<35} {'13.43':>8}")
    print(f"  {'NVFP4/FP8 RDQuant calibrated':<35} {'12.24':>8}")
    print(f"  {'AWQ+RDQuant INT4/INT8 (this run)':<35} {ppl:>8.2f}")
    print("-" * 50)

    return ppl


if __name__ == "__main__":
    main()
