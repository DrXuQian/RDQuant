"""
End-to-end AWQ + RDQuant INT4/INT8 → single Marlin kernel deployment.

Steps:
  1. Load Qwen3-4B and quantize with AWQ + RDQuant INT4/INT8
  2. Convert Int4FusedLinear → Int4MarlinLinear (single Marlin launch)
  3. Benchmark: quantized Marlin vs BF16 (prefill + decode)
  4. Verify PPL correctness on WikiText-2

Usage:
    python -m rdquant.int4_marlin_e2e
"""

from __future__ import annotations

import gc
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def convert_to_marlin(qmodel) -> nn.Module:
    """Convert all Int4FusedLinear layers to Int4MarlinLinear in-place.

    Returns the inner model (unwrapped from QuantizedModelInt4).
    """
    from rdquant.int4_fusion import Int4FusedLinear
    from rdquant.int4_marlin import Int4MarlinLinear

    model = qmodel.model
    converted = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, Int4FusedLinear):
            # Use exact stored raw weights and scales
            w_int4 = module.w_int4_raw.cpu()       # [N_int4, K] int8
            s_int4 = module.s_int4_raw.cpu()        # [N_int4, K//gs] float32
            w_int8 = module.w_int8_raw.cpu()        # [N_int8, K] int8
            s_int8 = module.s_int8_raw.cpu()        # [N_int8] float32
            inv_perm = module.inv_perm.cpu()
            group_size = module.group_size

            # Convert INT4 to UINT4
            w_int4_uint4 = (w_int4.to(torch.int16) + 8).to(torch.uint8)

            awq = module.awq_scales.cpu().float() if module.awq_scales is not None else None
            bias = module.bias.data.cpu() if module.bias is not None else None

            marlin_layer = Int4MarlinLinear(
                w_int4_uint4=w_int4_uint4,
                s_int4=s_int4,
                w_int8=w_int8,
                s_int8=s_int8,
                inv_perm=inv_perm,
                bias=bias,
                group_size=group_size,
                awq_scales=awq,
            )

            # Replace in model
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], marlin_layer)
            converted += 1

    print(f"  Converted {converted} layers to Int4MarlinLinear")
    return model


def benchmark_model(model, tokenizer, label: str, seq_lengths=[1, 128, 512],
                    n_warmup=3, n_iters=10):
    """Benchmark model inference latency at various sequence lengths."""
    device = next(model.parameters()).device
    model.eval()

    print(f"\n{'='*60}")
    print(f"Latency benchmark: {label}")
    print(f"{'='*60}")
    print(f"  {'SeqLen':>8} {'Latency(ms)':>12} {'Tok/s':>10}")
    print(f"  {'-'*8} {'-'*12} {'-'*10}")

    for seq_len in seq_lengths:
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                model(input_ids)
        torch.cuda.synchronize()

        # Timed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            for _ in range(n_iters):
                model(input_ids)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) / n_iters * 1000
        toks_per_s = seq_len / (elapsed_ms / 1000)

        print(f"  {seq_len:>8} {elapsed_ms:>12.2f} {toks_per_s:>10.0f}")

    return elapsed_ms  # return last


def benchmark_decode(model, tokenizer, label: str, n_tokens=50, n_warmup=2, n_iters=5):
    """Benchmark autoregressive decode latency."""
    device = next(model.parameters()).device
    model.eval()

    prompt_ids = torch.randint(0, tokenizer.vocab_size, (1, 16), device=device)

    print(f"\n  Decode ({n_tokens} tokens):")

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            ids = prompt_ids.clone()
            past = None
            for _ in range(n_tokens):
                out = model(ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[:, -1:].argmax(dim=-1)
                ids = next_tok

    torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_iters):
            ids = prompt_ids.clone()
            past = None
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n_tokens):
                out = model(ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_tok = out.logits[:, -1:].argmax(dim=-1)
                ids = next_tok
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

    avg_s = sum(times) / len(times)
    tok_s = n_tokens / avg_s
    ms_per_tok = avg_s / n_tokens * 1000
    print(f"    Total: {avg_s*1000:.1f}ms  |  {tok_s:.1f} tok/s  |  {ms_per_tok:.2f} ms/tok")
    return tok_s


def benchmark_decode_cuda_graph(model, tokenizer, label: str, n_tokens=50):
    """Benchmark single decode step with CUDA Graph."""
    device = next(model.parameters()).device
    model.eval()

    prompt_ids = torch.randint(0, tokenizer.vocab_size, (1, 16), device=device)

    # Build KV cache
    with torch.no_grad():
        ids = prompt_ids.clone()
        past = None
        out = model(ids, past_key_values=past, use_cache=True)
        past = out.past_key_values
        for _ in range(n_tokens - 1):
            next_tok = out.logits[:, -1:].argmax(dim=-1)
            out = model(next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values

    static_input = torch.zeros(1, 1, dtype=torch.long, device=device)
    static_input.copy_(out.logits[:, -1:].argmax(dim=-1))

    # Warmup for capture
    for _ in range(3):
        with torch.no_grad():
            _ = model(static_input, past_key_values=past, use_cache=True)
    torch.cuda.synchronize()

    # Capture CUDA Graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = model(static_input, past_key_values=past, use_cache=True)
    torch.cuda.synchronize()

    # Warmup replays
    for _ in range(50):
        g.replay()
    torch.cuda.synchronize()

    # Timed replays
    t0 = time.perf_counter()
    for _ in range(200):
        g.replay()
    torch.cuda.synchronize()
    graph_ms = (time.perf_counter() - t0) / 200 * 1000

    tok_s = 1000.0 / graph_ms
    print(f"  {label} CUDA Graph: {graph_ms:.2f} ms/tok → {tok_s:.0f} tok/s")
    return tok_s


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "/root/autodl-tmp/Qwen3-4B"
    wikitext_path = "/root/data/datasets/wikitext/test.parquet"

    print("=" * 70)
    print("AWQ + RDQuant INT4/INT8 → Single Marlin Kernel E2E Deployment")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    print("\n[1/7] Loading Qwen3-4B...")
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
    # 2. BF16 baseline benchmark
    # ------------------------------------------------------------------
    print("\n[2/7] BF16 baseline benchmarks...")
    benchmark_model(model, tokenizer, "BF16 Baseline", seq_lengths=[1, 128, 512])
    bf16_decode_tps = benchmark_decode(model, tokenizer, "BF16 Baseline", n_tokens=50)
    bf16_graph_tps = benchmark_decode_cuda_graph(model, tokenizer, "BF16")

    # ------------------------------------------------------------------
    # 3. Compute AWQ scales
    # ------------------------------------------------------------------
    print("\n[3/7] Computing AWQ scales...")
    t0 = time.time()

    calib_texts = [
        "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings. The seeds of modern AI were planted by philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols.",
        "In mathematics, a function is a relation between a set of inputs and a set of permissible outputs. Functions have the property that each input is related to exactly one output. The notation f(x) is commonly used to denote a function.",
    ]

    from rdquant.awq_scale import compute_awq_scales

    ignore_patterns = ["*lm_head*", "*embed*", "*norm*"]
    awq_scales = compute_awq_scales(
        model=model, tokenizer=tokenizer, calib_texts=calib_texts,
        max_samples=2, seq_length=512, ignore=ignore_patterns, device="cuda",
    )
    print(f"      Computed scales for {len(awq_scales)} layers in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------------
    # 4. Quantize with RDQuant INT4/INT8
    # ------------------------------------------------------------------
    print("\n[4/7] Quantizing with RDQuant INT4/INT8 at 5.3 bpw...")
    t0 = time.time()

    from rdquant.int4_quant import quantize_model_int4

    model.cpu()
    torch.cuda.empty_cache()
    gc.collect()

    qmodel = quantize_model_int4(
        model=model, budget_avg_bits=5.3, formats=["INT4", "INT8"],
        ignore=ignore_patterns, awq_scales=awq_scales, int4_group_size=128,
    )
    print(f"      Quantized in {time.time() - t0:.1f}s")

    gc.collect()
    torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # 5. Convert to Marlin and benchmark
    # ------------------------------------------------------------------
    print("\n[5/7] Converting to Int4MarlinLinear (single Marlin kernel)...")
    t0 = time.time()

    # Move to CUDA for Marlin conversion
    qmodel.cuda()
    marlin_model = convert_to_marlin(qmodel)
    marlin_model.cuda()
    print(f"      Converted in {time.time() - t0:.1f}s")

    gc.collect()
    torch.cuda.empty_cache()

    # Benchmark Marlin model
    benchmark_model(marlin_model, tokenizer, "INT4/INT8 Marlin (1 kernel)", seq_lengths=[1, 128, 512])
    marlin_decode_tps = benchmark_decode(marlin_model, tokenizer, "INT4/INT8 Marlin", n_tokens=50)
    print("\n[6/7] CUDA Graph decode benchmarks...")
    marlin_graph_tps = benchmark_decode_cuda_graph(marlin_model, tokenizer, "Marlin INT4/INT8")

    # ------------------------------------------------------------------
    # 7. PPL verification
    # ------------------------------------------------------------------
    print("\n[7/7] Verifying WikiText-2 PPL (Marlin model)...")
    t0 = time.time()

    from rdquant.eval import eval_perplexity

    ppl = eval_perplexity(
        model=marlin_model, tokenizer=tokenizer,
        dataset="wikitext", dataset_path=wikitext_path, seq_length=2048,
    )
    print(f"      WikiText-2 PPL: {ppl:.2f}  (computed in {time.time() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Decode (no CUDA Graph):")
    print(f"    BF16:   {bf16_decode_tps:.1f} tok/s")
    print(f"    Marlin: {marlin_decode_tps:.1f} tok/s")
    speedup = marlin_decode_tps / bf16_decode_tps if bf16_decode_tps > 0 else 0
    print(f"    Ratio:  {speedup:.2f}x")
    print(f"\n  Decode (CUDA Graph):")
    print(f"    BF16:   {bf16_graph_tps:.0f} tok/s  ({1000/bf16_graph_tps:.2f} ms/tok)")
    print(f"    Marlin: {marlin_graph_tps:.0f} tok/s  ({1000/marlin_graph_tps:.2f} ms/tok)")
    graph_speedup = marlin_graph_tps / bf16_graph_tps if bf16_graph_tps > 0 else 0
    print(f"    Ratio:  {graph_speedup:.2f}x")
    print(f"\n  Quality:")
    print(f"    WikiText-2 PPL: {ppl:.2f}")
    print(f"    BF16 reference: 12.90")
    print(f"    Avg bits/weight: 5.3 bpw (3.0x compression)")
    print("=" * 70)

    return ppl


if __name__ == "__main__":
    main()
