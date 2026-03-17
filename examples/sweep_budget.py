"""
Sweep average bits from 4 to 8, print R-D table, optionally save plot.

Usage:
    python examples/sweep_budget.py --model sshleifer/tiny-gpt2 --budgets 4 5 6 7 8
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep bit budgets for RDQuant and display R-D tradeoff."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sshleifer/tiny-gpt2",
        help="HuggingFace model ID or local path.",
    )
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="+",
        default=[4.0, 5.0, 6.0, 7.0, 8.0],
        help="List of bit budgets to sweep (default: 4 5 6 7 8).",
    )
    parser.add_argument(
        "--formats",
        type=str,
        nargs="+",
        default=["NVFP4", "MXFP6", "MXFP8", "FP16"],
        help="Allowed quantization formats.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use (default: cpu).",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional path to save a PNG plot of the R-D curve (requires matplotlib).",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate perplexity at each budget point (slow, requires datasets).",
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

    import copy
    import torch
    from rdquant.quantize import quantize_model

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    base_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32)
    base_model = base_model.to(args.device)

    print(f"\nSweeping budgets: {args.budgets}")
    print(f"Formats: {args.formats}")

    header = f"{'Budget':>8}  {'Actual bits':>12}  {'Total distortion':>18}  {'PPL':>8}"
    print("\n" + header)
    print("-" * len(header))

    results_table = []

    for budget in args.budgets:
        # Deep copy the original model weights each time (quantize_model modifies in place)
        model_copy = copy.deepcopy(base_model)
        qmodel = quantize_model(
            model_copy,
            budget_avg_bits=budget,
            formats=args.formats,
        )

        # Compute overall avg bits and total distortion
        total_bits = 0.0
        total_params = 0
        total_distortion = 0.0
        for layer_name, result in qmodel.layer_info.items():
            n_out = sum(result.splits.values())
            layer_bits = sum(s["total_bits"] for s in result.format_stats.values())
            if n_out > 0 and result.avg_bits > 0:
                n_in = int(round(layer_bits / (n_out * result.avg_bits)))
            else:
                n_in = 0
            total_params += n_out * n_in
            total_bits += layer_bits
            total_distortion += result.total_distortion

        actual_bits = total_bits / total_params if total_params > 0 else 0.0

        ppl_str = "N/A"
        ppl_val = float("nan")
        if args.eval:
            try:
                from rdquant.eval import eval_perplexity
                ppl_val = eval_perplexity(qmodel, tokenizer, dataset="wikitext", seq_length=512)
                ppl_str = f"{ppl_val:.2f}"
            except Exception as exc:
                ppl_str = f"err: {exc}"

        print(f"{budget:>8.1f}  {actual_bits:>12.3f}  {total_distortion:>18.6f}  {ppl_str:>8}")
        results_table.append({
            "budget": budget,
            "actual_bits": actual_bits,
            "total_distortion": total_distortion,
            "ppl": ppl_val,
        })

    if args.plot:
        try:
            import matplotlib.pyplot as plt

            bits_vals = [r["actual_bits"] for r in results_table]
            dist_vals = [r["total_distortion"] for r in results_table]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            axes[0].plot(bits_vals, dist_vals, "o-", color="steelblue")
            axes[0].set_xlabel("Average bits per element")
            axes[0].set_ylabel("Total distortion (sum MSE)")
            axes[0].set_title("Rate-Distortion Curve")
            axes[0].grid(True, alpha=0.3)

            if args.eval and any(r["ppl"] == r["ppl"] for r in results_table):  # any non-NaN
                import math
                ppl_vals = [r["ppl"] for r in results_table if math.isfinite(r["ppl"])]
                bits_for_ppl = [r["actual_bits"] for r in results_table if math.isfinite(r["ppl"])]
                axes[1].plot(bits_for_ppl, ppl_vals, "o-", color="crimson")
                axes[1].set_xlabel("Average bits per element")
                axes[1].set_ylabel("Perplexity")
                axes[1].set_title("Bits vs Perplexity")
                axes[1].grid(True, alpha=0.3)
            else:
                axes[1].set_visible(False)

            plt.tight_layout()
            plt.savefig(args.plot, dpi=150)
            print(f"\nPlot saved to: {args.plot}")
        except ImportError:
            print("\nmatplotlib is not installed; skipping plot. pip install matplotlib")


if __name__ == "__main__":
    main()
