"""
Save/load packed INT4/INT8 mixed-precision checkpoints in safetensors format.

Packed format per layer:
  - weight_int4:     [N_int4, K] int8 (values -8..7)
  - scale_int4:      [N_int4, K//group_size] float16
  - weight_int8:     [N_int8, K] int8 (values -128..127)
  - scale_int8:      [N_int8] float16
  - inv_perm:        [N_total] int32
  - awq_scales:      [K] float16 (optional)
  - bias:            [N_total] float16 (optional)
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Optional

import torch


def save_packed_int4(
    model,           # QuantizedModelInt4 or model with Int4FusedLinear layers
    path: str,
    source_model_dir: Optional[str] = None,
) -> None:
    """Save INT4/INT8 quantized model as packed safetensors.

    Args:
        model: Model with Int4FusedLinear layers (e.g. QuantizedModelInt4).
        path: Output directory.
        source_model_dir: Original HF model dir to copy tokenizer/config from.
    """
    from safetensors.torch import save_file
    from rdquant.int4_fusion import Int4FusedLinear

    os.makedirs(path, exist_ok=True)

    # Get inner model if wrapped
    inner = model.model if hasattr(model, 'model') else model

    tensors: dict[str, torch.Tensor] = {}
    layer_configs: dict[str, dict] = {}

    for name, module in inner.named_modules():
        if not isinstance(module, Int4FusedLinear):
            continue

        prefix = name

        # Store raw quantized weights
        tensors[f"{prefix}.weight_int4"] = module.w_int4_raw.to(torch.int8).contiguous()
        tensors[f"{prefix}.scale_int4"] = module.s_int4_raw.to(torch.float16).contiguous()
        tensors[f"{prefix}.weight_int8"] = module.w_int8_raw.to(torch.int8).contiguous()
        tensors[f"{prefix}.scale_int8"] = module.s_int8_raw.to(torch.float16).contiguous()
        tensors[f"{prefix}.inv_perm"] = module.inv_perm.to(torch.int32).contiguous()

        if module.awq_scales is not None:
            tensors[f"{prefix}.awq_scales"] = module.awq_scales.to(torch.float16).contiguous()

        if module.bias is not None:
            tensors[f"{prefix}.bias"] = module.bias.data.to(torch.float16).contiguous()

        layer_configs[name] = {
            "in_features": module.K,
            "out_features": module.N_int4 + module.N_int8,
            "n_int4": module.N_int4,
            "n_int8": module.N_int8,
            "group_size": module.group_size,
            "has_awq": module.awq_scales is not None,
            "has_bias": module.bias is not None,
        }

    # Also save non-quantized parameters (embed, lm_head, norms)
    for name, param in inner.named_parameters():
        # Skip if already saved as part of Int4FusedLinear
        skip = False
        for ln in layer_configs:
            if name.startswith(ln + "."):
                skip = True
                break
        if not skip:
            tensors[name] = param.data.to(torch.float16).contiguous()

    # Save non-quantized buffers (e.g. rotary embedding inv_freq)
    for name, buf in inner.named_buffers():
        skip = False
        for ln in layer_configs:
            if name.startswith(ln + "."):
                skip = True
                break
        if not skip and name not in tensors:
            tensors[name] = buf.contiguous()

    # Write safetensors
    save_file(tensors, os.path.join(path, "model.safetensors"))

    # Write quantization config
    budget = model.budget_avg_bits if hasattr(model, 'budget_avg_bits') else 5.3
    config = {
        "quant_method": "rdquant_int4",
        "formats": ["INT4", "INT8"],
        "budget_avg_bits": budget,
        "group_size": 128,
        "layers": layer_configs,
    }
    with open(os.path.join(path, "quantization_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer and config from source model
    if source_model_dir and os.path.isdir(source_model_dir):
        for fname in ["config.json", "generation_config.json",
                       "tokenizer.json", "tokenizer_config.json",
                       "vocab.json", "merges.txt",
                       "added_tokens.json", "special_tokens_map.json",
                       "chat_template.jinja"]:
            src = os.path.join(source_model_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(path, fname))

    # Compute total size
    total_bytes = sum(t.numel() * t.element_size() for t in tensors.values())
    n_layers = len(layer_configs)
    print(f"Saved {n_layers} quantized layers to {path}")
    print(f"  model.safetensors: {total_bytes / 1e9:.2f} GB")
    print(f"  Quantized layers: {n_layers}")


def load_packed_int4(
    path: str,
    model_class=None,
    device: str = "cpu",
):
    """Load packed INT4/INT8 checkpoint and reconstruct Int4FusedLinear layers.

    Args:
        path: Directory containing model.safetensors + quantization_config.json.
        model_class: Optional HF model class. If None, loads architecture from config.
        device: Device to load tensors to.

    Returns:
        Model with Int4FusedLinear layers.
    """
    from safetensors.torch import load_file
    from rdquant.int4_fusion import Int4FusedLinear

    # Load config
    with open(os.path.join(path, "quantization_config.json")) as f:
        qconfig = json.load(f)

    # Load tensors
    tensors = load_file(os.path.join(path, "model.safetensors"), device=device)

    # Load base model architecture
    if model_class is None:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.float16, device_map=device,
            trust_remote_code=True,
        )
    else:
        model = model_class.from_pretrained(
            path, torch_dtype=torch.float16, device_map=device,
            trust_remote_code=True,
        )

    # Replace linear layers with Int4FusedLinear
    for name, lcfg in qconfig["layers"].items():
        prefix = name

        w_int4 = tensors[f"{prefix}.weight_int4"]
        s_int4 = tensors[f"{prefix}.scale_int4"].float()
        w_int8 = tensors[f"{prefix}.weight_int8"]
        s_int8 = tensors[f"{prefix}.scale_int8"].float()
        inv_perm = tensors[f"{prefix}.inv_perm"].long()

        awq = tensors.get(f"{prefix}.awq_scales")
        if awq is not None:
            awq = awq.float()

        bias = tensors.get(f"{prefix}.bias")
        if bias is not None:
            bias = bias.float()

        qlayer = Int4FusedLinear(
            w_int4=w_int4,
            s_int4=s_int4,
            w_int8=w_int8,
            s_int8=s_int8,
            inv_perm=inv_perm,
            bias=bias,
            group_size=lcfg["group_size"],
            awq_scales=awq,
        )

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], qlayer)

    return model


def load_for_inference_int4(
    path: str,
    device: str = "cuda",
    use_marlin: bool = True,
) -> torch.nn.Module:
    """Load packed INT4/INT8 checkpoint with Marlin kernels for fast inference.

    Pipeline: safetensors → Int4FusedLinear → Int4MarlinLinear (single Marlin kernel).

    Args:
        path: Directory with model.safetensors + quantization_config.json.
        device: Target device.
        use_marlin: If True, convert to Int4MarlinLinear for Marlin inference.
            If False, keep Int4FusedLinear (fake-quant, slower but no vLLM needed).

    Returns:
        Model ready for inference.
    """
    from safetensors.torch import load_file
    from rdquant.int4_fusion import Int4FusedLinear

    # Load config
    with open(os.path.join(path, "quantization_config.json")) as f:
        qconfig = json.load(f)

    # Load all tensors
    tensors = load_file(os.path.join(path, "model.safetensors"), device="cpu")

    # Load HF model architecture (empty weights)
    from transformers import AutoModelForCausalLM, AutoConfig
    config = AutoConfig.from_pretrained(path, trust_remote_code=True)
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = model.to_empty(device="cpu")
    model.eval()

    # Fill non-quantized parameters (embed, norm, lm_head)
    quant_layers = set(qconfig["layers"].keys())
    filled = 0
    for pname, param in model.named_parameters():
        is_quant = any(pname.startswith(ql + ".") for ql in quant_layers)
        if not is_quant and pname in tensors:
            param.data.copy_(tensors[pname].to(param.dtype))
            filled += 1
    for bname, buf in model.named_buffers():
        is_quant = any(bname.startswith(ql + ".") for ql in quant_layers)
        if not is_quant and bname in tensors:
            buf.copy_(tensors[bname].to(buf.dtype))
            filled += 1

    # Restore weight tying (to_empty breaks it)
    if getattr(config, "tie_word_embeddings", False):
        if hasattr(model, "lm_head") and hasattr(model.model, "embed_tokens"):
            model.lm_head.weight = model.model.embed_tokens.weight
            filled += 1

    print(f"  Filled {filled} non-quantized parameters")

    # Replace quantized layers with Int4FusedLinear
    for name, lcfg in qconfig["layers"].items():
        prefix = name
        w_int4 = tensors[f"{prefix}.weight_int4"]
        s_int4 = tensors[f"{prefix}.scale_int4"].float()
        w_int8 = tensors[f"{prefix}.weight_int8"]
        s_int8 = tensors[f"{prefix}.scale_int8"].float()
        inv_perm = tensors[f"{prefix}.inv_perm"].long()

        awq = tensors.get(f"{prefix}.awq_scales")
        if awq is not None:
            awq = awq.float()
        bias = tensors.get(f"{prefix}.bias")
        if bias is not None:
            bias = bias.float()

        qlayer = Int4FusedLinear(
            w_int4=w_int4, s_int4=s_int4,
            w_int8=w_int8, s_int8=s_int8,
            inv_perm=inv_perm, bias=bias,
            group_size=lcfg["group_size"], awq_scales=awq,
        )

        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], qlayer)

    # Convert to Marlin if requested
    if use_marlin and device == "cuda" and torch.cuda.is_available():
        from rdquant.int4_marlin import Int4MarlinLinear

        model.to(device)
        converted = 0
        for name, module in list(model.named_modules()):
            if isinstance(module, Int4FusedLinear):
                w_int4 = module.w_int4_raw.cpu()
                s_int4 = module.s_int4_raw.cpu()
                w_int8 = module.w_int8_raw.cpu()
                s_int8 = module.s_int8_raw.cpu()
                inv_perm = module.inv_perm.cpu()
                group_size = module.group_size

                w_int4_uint4 = (w_int4.to(torch.int16) + 8).to(torch.uint8)
                awq = module.awq_scales.cpu().float() if module.awq_scales is not None else None
                bias = module.bias.data.cpu() if module.bias is not None else None

                marlin_layer = Int4MarlinLinear(
                    w_int4_uint4=w_int4_uint4, s_int4=s_int4,
                    w_int8=w_int8, s_int8=s_int8,
                    inv_perm=inv_perm, bias=bias,
                    group_size=group_size, awq_scales=awq,
                )

                parts = name.split(".")
                parent = model
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                setattr(parent, parts[-1], marlin_layer)
                converted += 1

        print(f"Loaded {converted} Int4MarlinLinear layers (single Marlin kernel)")
    else:
        model.to(device)
        print(f"Loaded {len(qconfig['layers'])} Int4FusedLinear layers (fake-quant)")

    return model
