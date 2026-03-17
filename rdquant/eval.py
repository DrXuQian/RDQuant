"""
Evaluation utilities for quantized models.

Provides perplexity computation (sliding-window) and zero-shot evaluation
via lm-eval-harness (gracefully skipped when not installed).
"""

from __future__ import annotations

from typing import Optional

import torch


def eval_perplexity(
    model,
    tokenizer,
    dataset: str = "wikitext",
    max_samples: Optional[int] = None,
    seq_length: int = 2048,
) -> float:
    """Compute perplexity of a model on a text dataset using a sliding window.

    Args:
        model: A :class:`~rdquant.quantize.QuantizedModel` or regular
            ``PreTrainedModel``. Must support ``model(input_ids)`` and
            return a ``CausalLMOutputWithCrossAttentions``-like object
            with a ``logits`` attribute.
        tokenizer: HuggingFace tokenizer compatible with the model.
        dataset: Dataset name to load from HuggingFace datasets.
            Supported: ``"wikitext"`` (loads ``wikitext-2-raw-v1``),
            ``"ptb"`` (loads ``ptb_text_only``). Defaults to ``"wikitext"``.
        max_samples: Maximum number of tokens to evaluate. If ``None``,
            evaluates the full test split.
        seq_length: Sliding window / context length. Defaults to 2048.

    Returns:
        Perplexity as a Python float.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for eval_perplexity. "
            "Install it with: pip install datasets"
        ) from exc

    device = next(model.parameters()).device

    # Load dataset
    if dataset == "wikitext":
        raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(raw["text"])
    elif dataset == "ptb":
        raw = load_dataset("ptb_text_only", "penn_treebank", split="test")
        text = "\n\n".join(raw["sentence"])
    else:
        # Generic: try loading as dataset_name with default config
        raw = load_dataset(dataset, split="test")
        # Try common text field names
        text_field = "text" if "text" in raw.column_names else raw.column_names[0]
        text = "\n\n".join(str(t) for t in raw[text_field])

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings["input_ids"].squeeze(0)  # [T]

    if max_samples is not None:
        input_ids = input_ids[:max_samples]

    total_tokens = input_ids.shape[0]
    nlls = []

    model.eval()
    with torch.no_grad():
        for begin in range(0, total_tokens - 1, seq_length):
            end = min(begin + seq_length, total_tokens)
            chunk = input_ids[begin:end].unsqueeze(0).to(device)  # [1, L]

            if chunk.shape[1] < 2:
                continue

            outputs = model(chunk, labels=chunk)
            # HuggingFace models return loss when labels are provided
            if hasattr(outputs, "loss") and outputs.loss is not None:
                nll = outputs.loss.item() * (chunk.shape[1] - 1)
            else:
                # Manual NLL computation from logits
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = chunk[..., 1:].contiguous()
                import torch.nn.functional as F
                nll = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="sum",
                ).item()
            nlls.append(nll)

    total_nll = sum(nlls)
    n_tokens = total_tokens - 1
    ppl = torch.exp(torch.tensor(total_nll / n_tokens)).item()
    return ppl


def eval_zero_shot(
    model,
    tokenizer,
    tasks: list[str] = None,
    num_fewshot: int = 0,
) -> dict[str, float]:
    """Run zero-shot evaluation using lm-eval-harness.

    Gracefully skips evaluation and returns an empty dict if ``lm_eval``
    is not installed.

    Args:
        model: A :class:`~rdquant.quantize.QuantizedModel` or regular
            ``PreTrainedModel``.
        tokenizer: HuggingFace tokenizer.
        tasks: List of task names to evaluate. Defaults to
            ``["arc_easy", "arc_challenge", "hellaswag", "mmlu", "winogrande"]``.
        num_fewshot: Number of few-shot examples. Defaults to 0 (zero-shot).

    Returns:
        Dict mapping task name -> accuracy (float in [0, 1]).
        Returns an empty dict if lm_eval is not installed.
    """
    if tasks is None:
        tasks = ["arc_easy", "arc_challenge", "hellaswag", "mmlu", "winogrande"]

    try:
        import lm_eval
    except ImportError:
        import warnings
        warnings.warn(
            "lm-eval-harness is not installed; skipping zero-shot evaluation. "
            "Install with: pip install lm-eval",
            stacklevel=2,
        )
        return {}

    try:
        from lm_eval.models.huggingface import HFLM
        from lm_eval import evaluator

        lm = HFLM(pretrained=model, tokenizer=tokenizer)
        results = evaluator.simple_evaluate(
            model=lm,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size="auto",
        )
        # Extract accuracy from results
        scores: dict[str, float] = {}
        for task in tasks:
            if task in results["results"]:
                task_res = results["results"][task]
                # lm-eval stores acc under "acc" or "acc_norm"
                acc = task_res.get("acc_norm", task_res.get("acc", 0.0))
                scores[task] = float(acc)
        return scores
    except Exception as exc:
        import warnings
        warnings.warn(
            f"Zero-shot evaluation failed with error: {exc}",
            stacklevel=2,
        )
        return {}
