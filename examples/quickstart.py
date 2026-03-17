"""
Minimal example: quantize a small model and evaluate.

Usage:
    python examples/quickstart.py --model sshleifer/tiny-gpt2 \\
                                   --budget 5.3 \\
                                   --output ./tiny-gpt2-rdquant
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quantize a HuggingFace causal LM with RDQuant."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sshleifer/tiny-gpt2",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=5.3,
        help="Average bits per element budget (default: 5.3).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./rdquant-output",
        help="Output directory for the quantized model.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run quantization on (default: cpu).",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["NVFP4", "MXFP6", "MXFP8", "FP16"],
        help="Allowed quantization formats.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate perplexity on WikiText-2 after quantization.",
    )
    parser.add_argument(
        "--ignore",
        type=str,
        nargs="*",
        default=None,
        help="Layer name patterns to skip quantization (fnmatch style).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "transformers is required. Install with: pip install transformers"
        ) from exc

    import torch
    from rdquant.quantize import quantize_model

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    model = model.to(args.device)

    # Print original model size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Original model: {n_params:,} parameters ({n_params * 2 / 1e6:.1f} MB in FP16)")

    print(f"\nQuantizing with budget={args.budget} bits, formats={args.formats}")
    qmodel = quantize_model(
        model,
        budget_avg_bits=args.budget,
        formats=args.formats,
        ignore=args.ignore,
    )
    qmodel = qmodel.to(args.device)

    print("\nAllocation summary:")
    qmodel.print_summary()

    if args.eval:
        from rdquant.eval import eval_perplexity
        print("\nEvaluating perplexity on WikiText-2...")
        ppl = eval_perplexity(qmodel, tokenizer, dataset="wikitext", seq_length=512)
        print(f"WikiText-2 Perplexity: {ppl:.2f}")

    print(f"\nSaving quantized model to: {args.output}")
    qmodel.save_pretrained(args.output)
    print("Done.")


if __name__ == "__main__":
    main()
