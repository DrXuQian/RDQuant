"""
AWQ-style per-input-channel scaling computation.

Computes activation-aware scaling factors by collecting activation statistics
via forward hooks during calibration, then deriving optimal per-channel scales.

Reference: Lin et al., "AWQ: Activation-aware Weight Quantization" (2023).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


def compute_awq_scales(
    model: nn.Module,
    tokenizer,
    calib_texts: list[str],
    max_samples: int = 8,
    seq_length: int = 512,
    ignore: Optional[list[str]] = None,
    device: str = "cuda",
) -> dict[str, torch.Tensor]:
    """Compute per-input-channel AWQ scaling factors.

    For each linear layer, collects activation statistics via forward hooks:
      s_k = E[|x_k|] for each input channel k
    Then computes optimal scaling: alpha_k = s_k^0.5 (AWQ's default power)

    Args:
        model: The model to calibrate.
        tokenizer: HuggingFace tokenizer compatible with the model.
        calib_texts: List of calibration text strings.
        max_samples: Maximum number of calibration samples to use.
        seq_length: Maximum sequence length for tokenization.
        ignore: List of layer name patterns to skip.
        device: Device to run calibration on.

    Returns:
        Dict mapping layer_name -> alpha tensor of shape [K] (input channels).
    """
    if ignore is None:
        ignore = []

    # Collect activation magnitudes via forward hooks
    act_sums: dict[str, torch.Tensor] = {}
    act_counts: dict[str, int] = {}
    hook_handles = []

    def _should_ignore(name: str) -> bool:
        import fnmatch
        for pattern in ignore:
            if fnmatch.fnmatch(name, pattern) or pattern in name:
                return True
        return False

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and not _should_ignore(name):
            def hook(module, input, output, name=name):
                x = input[0].detach().float()
                # x shape: [..., K] -- flatten all but last dim
                x_2d = x.reshape(-1, x.shape[-1])
                mag = x_2d.abs().mean(dim=0)  # [K]
                if name in act_sums:
                    act_sums[name] = act_sums[name] + mag
                    act_counts[name] = act_counts[name] + 1
                else:
                    act_sums[name] = mag
                    act_counts[name] = 1

            handle = mod.register_forward_hook(hook)
            hook_handles.append(handle)

    # Run calibration forward passes
    model.eval()
    n_samples = min(max_samples, len(calib_texts))
    with torch.no_grad():
        for i in range(n_samples):
            text = calib_texts[i]
            tokens = tokenizer(
                text,
                return_tensors="pt",
                max_length=seq_length,
                truncation=True,
            ).input_ids.to(device)
            model(tokens)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    # Compute scales from averaged activation magnitudes
    awq_scales: dict[str, torch.Tensor] = {}
    for name, mag_sum in act_sums.items():
        count = act_counts[name]
        avg_mag = mag_sum / count  # [K] mean |activation|

        # AWQ default: alpha = activation_magnitude ^ 0.5
        alpha = avg_mag.pow(0.5)

        # Normalize so mean(alpha) = 1
        alpha_mean = alpha.mean()
        if alpha_mean > 0:
            alpha = alpha / alpha_mean

        # Clamp to avoid numerical issues
        alpha = alpha.clamp(min=1e-5)
        awq_scales[name] = alpha

    return awq_scales
