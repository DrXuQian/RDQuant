"""
Layer-wise importance calibration using a small amount of data.

Runs a forward (and optionally backward) pass on calibration tokens to
compute per-layer importance weights.  These weights rescale distortion
in the R-D allocator so that sensitive layers receive more bits.

Supported importance metrics:
  - "perturb":   Delta-loss when each layer is quantized to MXFP4 while others
                 stay BF16.  Most accurate, but O(n_layers) forward passes.
  - "fisher":    ||dL/dW||_F^2  (requires backward pass)
  - "act_norm":  ||X||_F^2 averaged over calibration tokens
                 WARNING: dominated by residual stream growth, not recommended.
  - "grad_norm": ||dL/dW||_F   (requires backward pass)

Usage::

    from rdquant.core.calibrate import compute_layer_importance

    importance = compute_layer_importance(
        model, tokenizer, calib_texts,
        metric="fisher", max_samples=8, seq_length=512,
    )
    # importance: dict[str, float]  — layer_name -> weight ≥ 0
    # Pass to quantize_model(..., layer_importance=importance)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def compute_layer_importance(
    model: nn.Module,
    tokenizer,
    calib_texts: list[str],
    metric: str = "fisher",
    max_samples: int = 8,
    seq_length: int = 512,
    ignore: list[str] | None = None,
    device: str | None = None,
) -> dict[str, float]:
    """Compute per-layer importance weights from calibration data.

    Args:
        model: The pretrained model (not yet quantized).
        tokenizer: HuggingFace tokenizer for encoding ``calib_texts``.
        calib_texts: List of calibration strings.
        metric: One of ``"fisher"``, ``"act_norm"``, ``"grad_norm"``.
        max_samples: Number of calibration samples to use.
        seq_length: Maximum sequence length per sample.
        ignore: Layer name patterns to skip (same as quantize_model ignore).
        device: Device to run calibration on.  Defaults to model's device.

    Returns:
        Dict mapping layer name (matching ``model.named_modules()``) to a
        non-negative importance weight.  Normalised so the mean weight is 1.0.
    """
    import fnmatch

    if device is None:
        device = next(model.parameters()).device

    ignore = ignore or []

    def _should_ignore(name: str) -> bool:
        for pat in ignore:
            if fnmatch.fnmatch(name, pat) or pat in name:
                return True
        return False

    # Collect target linear layers
    target_layers: dict[str, nn.Linear] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not _should_ignore(name):
            target_layers[name] = mod

    if not target_layers:
        return {}

    # Tokenize calibration data
    tokens_list = []
    for text in calib_texts[:max_samples]:
        enc = tokenizer(
            text, return_tensors="pt", max_length=seq_length,
            truncation=True, add_special_tokens=True,
        )
        tokens_list.append(enc["input_ids"].to(device))

    if not tokens_list:
        return {name: 1.0 for name in target_layers}

    model = model.to(device).eval()

    if metric == "perturb":
        return _compute_perturb_importance(model, target_layers, tokens_list)
    elif metric == "act_norm":
        return _compute_act_norm(model, target_layers, tokens_list)
    elif metric in ("fisher", "grad_norm"):
        return _compute_grad_importance(model, target_layers, tokens_list, metric)
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. "
            f"Choose from: perturb, fisher, act_norm, grad_norm"
        )


def _compute_perturb_importance(
    model: nn.Module,
    target_layers: dict[str, nn.Linear],
    tokens_list: list[torch.Tensor],
) -> dict[str, float]:
    """Compute importance = delta-loss when each layer is quantized to MXFP4.

    For each layer:
      1. Save original weight
      2. Replace with MXFP4 quantize→dequantize reconstruction
      3. Measure average loss across calibration tokens
      4. Restore original weight

    Importance = loss_perturbed - loss_baseline.  Higher = more sensitive.

    This requires O(n_layers) forward passes but is the most accurate metric.
    """
    from rdquant.core.formats import quantize, dequantize

    # Compute baseline loss
    baseline_loss = _avg_loss(model, tokens_list)

    raw: dict[str, float] = {}
    for name, mod in target_layers.items():
        # Save original weight
        orig_weight = mod.weight.data.clone()

        # Quantize weight to MXFP4 and dequantize
        with torch.no_grad():
            flat = orig_weight.flatten().float()
            qt = quantize(flat, "MXFP4")
            recon = dequantize(qt).reshape(orig_weight.shape).to(orig_weight.dtype)
            mod.weight.data.copy_(recon)

        # Measure perturbed loss
        perturbed_loss = _avg_loss(model, tokens_list)

        # Restore original weight
        mod.weight.data.copy_(orig_weight)

        delta = max(perturbed_loss - baseline_loss, 0.0)
        raw[name] = delta

    return _normalize(raw)


@torch.no_grad()
def _avg_loss(model: nn.Module, tokens_list: list[torch.Tensor]) -> float:
    """Average cross-entropy loss over calibration tokens."""
    model.eval()
    total_loss, n = 0.0, 0
    for toks in tokens_list:
        out = model(toks, labels=toks)
        loss = out.loss if hasattr(out, 'loss') else out[0]
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def _compute_act_norm(
    model: nn.Module,
    target_layers: dict[str, nn.Linear],
    tokens_list: list[torch.Tensor],
) -> dict[str, float]:
    """Compute importance = mean ||X_input||_F^2 per layer."""
    accum: dict[str, float] = {name: 0.0 for name in target_layers}
    counts: dict[str, int] = {name: 0 for name in target_layers}

    hooks = []

    def _make_hook(layer_name: str):
        def hook_fn(module, input, output):
            x = input[0].detach().float()
            accum[layer_name] += (x ** 2).sum().item()
            counts[layer_name] += x.numel()
        return hook_fn

    for name, mod in target_layers.items():
        hooks.append(mod.register_forward_hook(_make_hook(name)))

    with torch.no_grad():
        for toks in tokens_list:
            model(toks, labels=toks)

    for h in hooks:
        h.remove()

    # Mean squared activation per element
    raw = {name: accum[name] / max(counts[name], 1) for name in target_layers}
    return _normalize(raw)


def _compute_grad_importance(
    model: nn.Module,
    target_layers: dict[str, nn.Linear],
    tokens_list: list[torch.Tensor],
    metric: str,
) -> dict[str, float]:
    """Compute importance from gradient norms (fisher or grad_norm)."""
    accum: dict[str, float] = {name: 0.0 for name in target_layers}
    n_samples = 0

    model.train()  # need gradients
    for toks in tokens_list:
        model.zero_grad()
        out = model(toks, labels=toks)
        loss = out.loss if hasattr(out, 'loss') else out[0]
        loss.backward()

        for name, mod in target_layers.items():
            if mod.weight.grad is not None:
                g = mod.weight.grad.detach().float()
                if metric == "fisher":
                    accum[name] += (g ** 2).sum().item()
                else:  # grad_norm
                    accum[name] += g.norm().item()

        n_samples += 1

    model.eval()

    raw = {name: accum[name] / max(n_samples, 1) for name in target_layers}
    return _normalize(raw)


def _normalize(raw: dict[str, float]) -> dict[str, float]:
    """Normalise so that mean importance = 1.0."""
    vals = list(raw.values())
    if not vals:
        return raw
    mean_val = sum(vals) / len(vals)
    if mean_val <= 0:
        return {k: 1.0 for k in raw}
    return {k: v / mean_val for k, v in raw.items()}
