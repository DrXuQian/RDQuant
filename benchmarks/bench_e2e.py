"""End-to-end latency benchmark for RDQuant inference.

Compares decode and prefill latency between a BF16 baseline model and
an RDQuant Marlin-accelerated model loaded from a packed checkpoint.

Usage::

    python benchmarks/bench_e2e.py \\
        --checkpoint /path/to/packed \\
        --model /path/to/bf16

    # Skip BF16 baseline (only benchmark RDQuant):
    python benchmarks/bench_e2e.py --checkpoint /path/to/packed

    # Custom iterations:
    python benchmarks/bench_e2e.py --checkpoint /path/to/packed --model /path/to/bf16 \\
        --warmup 10 --iters 50 --seq-len 128
"""

from __future__ import annotations

import argparse
import sys
import time

import torch


def _cuda_timer(fn, warmup: int = 10, iters: int = 50) -> float:
    """Time *fn* using CUDA events.  Returns median latency in ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def _benchmark_model(
    model,
    tokenizer,
    device: str,
    seq_len: int,
    warmup: int,
    iters: int,
) -> dict[str, float]:
    """Run decode and prefill benchmarks, return latencies in ms."""
    # Build dummy input
    dummy_text = "Hello " * (seq_len // 2)
    enc = tokenizer(
        dummy_text, return_tensors="pt", max_length=seq_len,
        truncation=True, add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, device=device)
    else:
        attention_mask = attention_mask.to(device)

    # Prefill: single forward pass on seq_len tokens
    @torch.no_grad()
    def prefill_fn():
        model(input_ids, attention_mask=attention_mask, use_cache=True)

    prefill_ms = _cuda_timer(prefill_fn, warmup=warmup, iters=iters)

    # Decode: one cached autoregressive step after a prefill.
    with torch.no_grad():
        prefill_out = model(input_ids, attention_mask=attention_mask, use_cache=True)
    past_key_values = prefill_out.past_key_values
    decode_input_ids = input_ids[:, -1:].contiguous()
    decode_attention_mask = torch.ones(
        (input_ids.size(0), input_ids.size(1) + 1),
        device=device,
        dtype=attention_mask.dtype,
    )

    @torch.no_grad()
    def decode_fn():
        model(
            decode_input_ids,
            attention_mask=decode_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

    decode_ms = _cuda_timer(decode_fn, warmup=warmup, iters=iters)

    return {"prefill_ms": prefill_ms, "decode_ms": decode_ms}


def main():
    parser = argparse.ArgumentParser(description="RDQuant end-to-end latency benchmark")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to packed RDQuant checkpoint")
    parser.add_argument("--model", type=str, default=None,
                        help="Path to BF16 HuggingFace model (for baseline)")
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations (default: 10)")
    parser.add_argument("--iters", type=int, default=50,
                        help="Measurement iterations (default: 50)")
    parser.add_argument("--seq-len", type=int, default=128,
                        help="Sequence length for prefill (default: 128)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)

    results = {}

    # --- BF16 baseline ---
    if args.model is not None:
        print(f"Loading BF16 model from {args.model} ...")
        from transformers import AutoModelForCausalLM
        bf16_model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map=args.device,
        )
        bf16_model.eval()

        print("Benchmarking BF16 ...")
        results["BF16"] = _benchmark_model(
            bf16_model, tokenizer, args.device,
            seq_len=args.seq_len, warmup=args.warmup, iters=args.iters,
        )
        # Free memory
        del bf16_model
        torch.cuda.empty_cache()

    # --- RDQuant Marlin ---
    print(f"Loading RDQuant Marlin model from {args.checkpoint} ...")
    sys.path.insert(0, "/root/autodl-tmp/vllm_site")
    from rdquant.inference import load_for_inference
    rdq_model = load_for_inference(
        args.checkpoint, device=args.device, use_marlin=True,
    )
    rdq_model.eval()

    print("Benchmarking RDQuant Marlin ...")
    results["RDQuant-Marlin"] = _benchmark_model(
        rdq_model, tokenizer, args.device,
        seq_len=args.seq_len, warmup=args.warmup, iters=args.iters,
    )

    # --- Print comparison table ---
    print("\n" + "=" * 60)
    print(f"{'Benchmark':<20} {'Prefill (ms)':>14} {'Decode (ms)':>14}")
    print("-" * 60)
    for name, res in results.items():
        print(f"{name:<20} {res['prefill_ms']:>14.2f} {res['decode_ms']:>14.2f}")

    if "BF16" in results and "RDQuant-Marlin" in results:
        bf16 = results["BF16"]
        rdq = results["RDQuant-Marlin"]
        print("-" * 60)
        prefill_speedup = bf16["prefill_ms"] / rdq["prefill_ms"] if rdq["prefill_ms"] > 0 else float("inf")
        decode_speedup = bf16["decode_ms"] / rdq["decode_ms"] if rdq["decode_ms"] > 0 else float("inf")
        print(f"{'Speedup':<20} {prefill_speedup:>13.2f}x {decode_speedup:>13.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
