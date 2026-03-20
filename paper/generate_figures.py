#!/usr/bin/env python3
"""Generate all 6 figures for the RDQuant paper."""

import sys
sys.path.insert(0, '/root/rdquant')

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'figure.figsize': (6, 4),
    'figure.dpi': 150,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

OUTDIR = '/root/rdquant/paper/figures'
os.makedirs(OUTDIR, exist_ok=True)

MODEL_PATH = '/root/autodl-tmp/Qwen3-4B'

# ============================================================
# Load model once
# ============================================================
print("Loading model...")
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float32, device_map="cpu",
)
model.eval()
print("Model loaded.")

from rdquant.core.formats import compute_mse_2d, compute_mse

# ============================================================
# Precompute data needed by multiple figures
# ============================================================

# Collect all linear layers (excluding lm_head and embed_tokens)
linear_layers = {}
for name, mod in model.named_modules():
    if isinstance(mod, nn.Linear) and 'lm_head' not in name and 'embed_tokens' not in name:
        linear_layers[name] = mod

print(f"Found {len(linear_layers)} linear layers")

# ============================================================
# Fig 1: Channel-level MSE distribution
# ============================================================
print("Generating Fig 1: MSE distribution...")

target_layer = 'model.layers.20.self_attn.q_proj'
w = linear_layers[target_layer].weight.data.float()

with torch.inference_mode():
    mse_nvfp4 = compute_mse_2d(w, "NVFP4").numpy()
    mse_fp8 = compute_mse_2d(w, "FP8").numpy()

fig, ax = plt.subplots(figsize=(6, 4))
# Use log-spaced bins
all_vals = np.concatenate([mse_nvfp4[mse_nvfp4 > 0], mse_fp8[mse_fp8 > 0]])
bins = np.logspace(np.log10(all_vals.min() * 0.5), np.log10(all_vals.max() * 2), 60)

ax.hist(mse_nvfp4[mse_nvfp4 > 0], bins=bins, alpha=0.65, label='NVFP4 (4-bit)', color='#2196F3', edgecolor='white', linewidth=0.3)
ax.hist(mse_fp8[mse_fp8 > 0], bins=bins, alpha=0.65, label='FP8 (8-bit)', color='#FF9800', edgecolor='white', linewidth=0.3)
ax.set_xscale('log')
ax.set_xlabel('Per-channel quantization MSE')
ax.set_ylabel('Number of channels')
ax.set_title('Per-channel quantization MSE distribution\n(model.layers.20.self_attn.q_proj, 4096 channels)')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig1_mse_distribution.pdf'), bbox_inches='tight')
plt.close(fig)
print("  Saved fig1_mse_distribution.pdf")

# ============================================================
# Fig 2: R-D curve
# ============================================================
print("Generating Fig 2: R-D curve...")

data_points = [
    ('Uniform NVFP4', 4.00, 13.22, 's', '#2196F3', 60),
    ('RDQuant data-free', 5.52, 13.43, 'o', '#4CAF50', 60),
    ('RDQuant calibrated', 5.29, 12.24, '*', '#E91E63', 200),
    ('Uniform FP8', 8.13, 12.93, 's', '#FF9800', 60),
    ('BF16', 16.00, 12.90, 'D', '#9E9E9E', 60),
]

fig, ax = plt.subplots(figsize=(7, 4.5))

# Horizontal dashed line at BF16 PPL
ax.axhline(y=12.90, color='#9E9E9E', linestyle='--', alpha=0.5, linewidth=1, label='BF16 baseline')

# Connect RDQuant points with dashed line
rdq_x = [5.52, 5.29]
rdq_y = [13.43, 12.24]
ax.plot(rdq_x, rdq_y, '--', color='#4CAF50', alpha=0.5, linewidth=1.5)

# Plot each point
for label, x, y, marker, color, sz in data_points:
    ax.scatter(x, y, marker=marker, s=sz, color=color, zorder=5, edgecolors='black', linewidth=0.5)

# Annotate
offsets = {
    'Uniform NVFP4': (10, 10),
    'RDQuant data-free': (10, 8),
    'RDQuant calibrated': (10, -15),
    'Uniform FP8': (10, 8),
    'BF16': (-60, 10),
}
for label, x, y, marker, color, sz in data_points:
    ox, oy = offsets[label]
    ax.annotate(label, (x, y), textcoords='offset points', xytext=(ox, oy),
                fontsize=8, color=color,
                arrowprops=dict(arrowstyle='->', color=color, lw=0.8) if abs(ox) > 5 else None)

ax.set_xlabel('Average bits per weight')
ax.set_ylabel('WikiText-2 perplexity')
ax.set_title('Rate-Distortion curve (Qwen3-4B)')
ax.set_xlim(3, 17)
ax.set_ylim(11.5, 14.0)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig2_rd_curve.pdf'), bbox_inches='tight')
plt.close(fig)
print("  Saved fig2_rd_curve.pdf")

# ============================================================
# Fig 3: Per-layer allocation heatmap (data-free only)
# ============================================================
print("Generating Fig 3: Per-layer allocation...")

from rdquant import quantize_model
import copy

# We need a fresh copy for quantization since quantize_model modifies in-place
model_copy = copy.deepcopy(model)
qm = quantize_model(model_copy, budget_avg_bits=5.3, ignore=["lm_head", "embed_tokens"])

# Parse layer_info into per-transformer-layer, per-linear-type splits
# Layer names look like: model.layers.X.self_attn.{q,k,v,o}_proj, model.layers.X.mlp.{gate,up,down}_proj
layer_types = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
type_abbrev = {'q_proj': 'q', 'k_proj': 'k', 'v_proj': 'v', 'o_proj': 'o',
               'gate_proj': 'g', 'up_proj': 'u', 'down_proj': 'd'}

# Find number of transformer layers
n_transformer_layers = 0
for name in qm.layer_info:
    parts = name.split('.')
    for i, p in enumerate(parts):
        if p == 'layers' and i + 1 < len(parts):
            idx = int(parts[i + 1])
            n_transformer_layers = max(n_transformer_layers, idx + 1)

print(f"  Found {n_transformer_layers} transformer layers")

# Build allocation matrix: [n_transformer_layers * 7, 3] for NVFP4/FP8/FP16 percentages
formats = ['NVFP4', 'FP8', 'FP16']
n_cols = n_transformer_layers * len(layer_types)
pct_data = np.zeros((3, n_cols))  # 3 formats x n_cols
labels = []

for li in range(n_transformer_layers):
    for ti, lt in enumerate(layer_types):
        col_idx = li * len(layer_types) + ti
        labels.append(f'L{li}.{type_abbrev[lt]}')

        # Find matching layer_info entry
        found = False
        for name, result in qm.layer_info.items():
            if f'layers.{li}.' in name and lt in name:
                total_ch = sum(result.splits.values())
                if total_ch > 0:
                    for fi, fmt in enumerate(formats):
                        pct_data[fi, col_idx] = result.splits.get(fmt, 0) / total_ch * 100
                found = True
                break

fig, ax = plt.subplots(figsize=(16, 4))

x = np.arange(n_cols)
width = 0.8
colors = ['#2196F3', '#FF9800', '#4CAF50']

bottom = np.zeros(n_cols)
for fi, (fmt, color) in enumerate(zip(formats, colors)):
    ax.bar(x, pct_data[fi], width, bottom=bottom, color=color, label=fmt, edgecolor='none')
    bottom += pct_data[fi]

ax.set_ylabel('Channel allocation (%)')
ax.set_title(f'Per-layer format allocation (data-free, 5.3 bpw, Qwen3-4B)')
ax.set_xlim(-0.5, n_cols - 0.5)
ax.set_ylim(0, 100)

# Show every 7th label (one per transformer layer)
tick_positions = list(range(0, n_cols, 7))
tick_labels_short = [f'L{i}' for i in range(n_transformer_layers)]
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels_short, fontsize=7, rotation=45, ha='right')

ax.legend(loc='upper right')
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig3_allocation.pdf'), bbox_inches='tight')
plt.close(fig)
print("  Saved fig3_allocation.pdf")

# Free the copy
del model_copy, qm

# ============================================================
# Fig 4: Importance comparison (weight MSE + act_norm)
# ============================================================
print("Generating Fig 4: Importance comparison...")

# Weight MSE importance: sum of per-channel NVFP4 MSE for each layer
weight_mse_importance = {}
with torch.inference_mode():
    for name, mod in linear_layers.items():
        w = mod.weight.data.float()
        mse = compute_mse_2d(w, "NVFP4")
        weight_mse_importance[name] = mse.sum().item()

# Activation norm importance: need a forward pass
# Load tokenizer and run one sample
print("  Computing activation norms...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# Use a simple text for calibration
calib_text = "The quick brown fox jumps over the lazy dog. " * 20

device = 'cpu'
# Check if CUDA is available
if torch.cuda.is_available():
    device = 'cuda'
    model = model.to(device)

enc = tokenizer(calib_text, return_tensors='pt', max_length=256, truncation=True)
input_ids = enc['input_ids'].to(device)

act_norm_importance = {}
hooks = []

def make_hook(layer_name):
    def hook_fn(module, input, output):
        x = input[0].detach().float()
        act_norm_importance[layer_name] = (x ** 2).sum().item() / x.numel()
    return hook_fn

for name, mod in linear_layers.items():
    hooks.append(mod.register_forward_hook(make_hook(name)))

with torch.no_grad():
    model(input_ids)

for h in hooks:
    h.remove()

# Move model back to CPU to free GPU memory
if device == 'cuda':
    model = model.to('cpu')
    torch.cuda.empty_cache()

# Sort layers by their order in the model
layer_names_ordered = sorted(linear_layers.keys(), key=lambda n: list(linear_layers.keys()).index(n))
layer_indices = list(range(len(layer_names_ordered)))

wmse_vals = [weight_mse_importance[n] for n in layer_names_ordered]
anorm_vals = [act_norm_importance.get(n, 1e-12) for n in layer_names_ordered]

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(layer_indices, wmse_vals, '-', color='#2196F3', alpha=0.7, linewidth=0.8, label='Weight MSE (data-free)')
ax.plot(layer_indices, anorm_vals, '-', color='#E91E63', alpha=0.7, linewidth=0.8, label='Activation norm')
ax.set_yscale('log')
ax.set_xlabel('Layer index')
ax.set_ylabel('Importance (log scale)')
ax.set_title('Per-layer importance: Weight MSE vs Activation norm')
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig4_importance.pdf'), bbox_inches='tight')
plt.close(fig)
print("  Saved fig4_importance.pdf")

# ============================================================
# Fig 5: Inference latency bar chart
# ============================================================
print("Generating Fig 5: Latency bar chart...")

categories = ['Decode', 'Prefill']
bf16_vals = [32.59, 23.59]
eager_vals = [50.79, 45.17]
cuda_graph_vals = [12.57, 18.86]

x = np.arange(len(categories))
width = 0.22

fig, ax = plt.subplots(figsize=(6, 4))

bars_bf16 = ax.bar(x - width, bf16_vals, width, label='BF16', color='#9E9E9E', edgecolor='white')
bars_eager = ax.bar(x, eager_vals, width, label='RDQuant (Eager)', color='#2196F3', edgecolor='white')
bars_cuda = ax.bar(x + width, cuda_graph_vals, width, label='RDQuant (CUDA Graph)', color='#E91E63', edgecolor='white')

# Add speedup labels on CUDA Graph bars
speedups = [f'{bf16_vals[i]/cuda_graph_vals[i]:.2f}x' for i in range(len(categories))]
for bar, speedup in zip(bars_cuda, speedups):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            speedup, ha='center', va='bottom', fontsize=9, fontweight='bold', color='#E91E63')

ax.set_ylabel('Latency (ms)')
ax.set_title('Inference latency comparison (Qwen3-4B)')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
ax.set_ylim(0, max(eager_vals) * 1.15)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig5_latency.pdf'), bbox_inches='tight')
plt.close(fig)
print("  Saved fig5_latency.pdf")

# ============================================================
# Fig 6: Outlier ratio vs NVFP4 MSE scatter
# ============================================================
print("Generating Fig 6: Outlier scatter...")

# Pick representative layers of different types
target_layers_fig6 = {
    'q_proj (L5)': 'model.layers.5.self_attn.q_proj',
    'v_proj (L15)': 'model.layers.15.self_attn.v_proj',
    'gate_proj (L25)': 'model.layers.25.mlp.gate_proj',
    'down_proj (L30)': 'model.layers.30.mlp.down_proj',
}

fig, ax = plt.subplots(figsize=(6, 5))
colors_scatter = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']

with torch.inference_mode():
    for (label, layer_name), color in zip(target_layers_fig6.items(), colors_scatter):
        w = linear_layers[layer_name].weight.data.float()
        n_out = w.shape[0]

        # Per-channel outlier ratio: max|w_j| / std(w_j)
        abs_max = w.abs().max(dim=1).values
        std = w.std(dim=1).clamp(min=1e-12)
        outlier_ratio = (abs_max / std).numpy()

        # Per-channel NVFP4 MSE
        mse = compute_mse_2d(w, "NVFP4").numpy()

        # Filter out zero MSE
        mask = mse > 0
        ax.scatter(outlier_ratio[mask], mse[mask], s=8, alpha=0.4, color=color, label=label, edgecolors='none')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Outlier ratio (max|w| / std)')
ax.set_ylabel('NVFP4 quantization MSE')
ax.set_title('Channel outlier ratio vs quantization error')
ax.legend(markerscale=3)

# Add trend line (fit log-log regression on pooled data)
all_x, all_y = [], []
with torch.inference_mode():
    for label, layer_name in target_layers_fig6.items():
        w = linear_layers[layer_name].weight.data.float()
        abs_max = w.abs().max(dim=1).values
        std = w.std(dim=1).clamp(min=1e-12)
        outlier_ratio = (abs_max / std).numpy()
        mse = compute_mse_2d(w, "NVFP4").numpy()
        mask = mse > 0
        all_x.extend(outlier_ratio[mask].tolist())
        all_y.extend(mse[mask].tolist())

all_x = np.array(all_x)
all_y = np.array(all_y)
# Log-log linear fit
log_x = np.log10(all_x)
log_y = np.log10(all_y)
coeffs = np.polyfit(log_x, log_y, 1)
fit_x = np.logspace(np.log10(all_x.min()), np.log10(all_x.max()), 100)
fit_y = 10 ** (coeffs[0] * np.log10(fit_x) + coeffs[1])
ax.plot(fit_x, fit_y, '--', color='black', alpha=0.5, linewidth=1.5,
        label=f'Trend (slope={coeffs[0]:.1f})')
ax.legend(markerscale=3, fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, 'fig6_outlier_scatter.pdf'), bbox_inches='tight')
plt.close(fig)
print("  Saved fig6_outlier_scatter.pdf")

print("\nAll 6 figures generated successfully!")
print(f"Output directory: {OUTDIR}")
for f in sorted(os.listdir(OUTDIR)):
    if f.endswith('.pdf'):
        print(f"  {f}")
