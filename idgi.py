
from __future__ import annotations

import time
import torch
import torch.nn as nn

from utility import (
    AttributionResult,
    _straight_line_pass,
    _pack_result,
    _rescale,
)


def compute_idgi(
    model: nn.Module,
    input: torch.Tensor,
    params: dict,
) -> AttributionResult:
    """
    Compute IDGI attribution.

    Args:
        model: PyTorch model that outputs scalar logits (use ClassLogitModel wrapper)
        input: Input tensor (1, C, H, W)
        params: Dictionary with:
            - baseline: Baseline tensor (1, C, H, W)
            - N: Number of interpolation steps (default: 50)

    Returns:
        AttributionResult containing attributions and metrics
    """
    baseline = params['baseline']
    N = params.get('N', 50)

    t0 = time.time()

    # Compute gradients along straight line
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, input, baseline, N)

    grad_stack = torch.cat(grads, dim=0)                    # (N, C, H, W)
    df_arr = torch.tensor(df_list, device=input.device)     # (N,)

    # ‖g_k‖² per step
    g_norm_sq = grad_stack.view(N, -1).pow(2).sum(dim=1)    # (N,)
    safe_norm = torch.where(g_norm_sq > 1e-12, g_norm_sq, torch.ones_like(g_norm_sq))

    # d_k / ‖g_k‖²  →  (N, 1, 1, 1)
    scale = (df_arr / safe_norm).view(N, 1, 1, 1)

    # Σ_k  g_i² * d_k / ‖g_k‖²
    # Zero out steps where ‖g‖² was too small
    mask = (g_norm_sq > 1e-12).view(N, 1, 1, 1)
    attr = (grad_stack.pow(2) * scale * mask).sum(dim=0, keepdim=True)  # (1, C, H, W)

    attr = _rescale(attr, target)

    # μ_k ∝ |Δf_k| for diagnostics
    weights = df_arr.abs()
    w_sum = weights.sum()
    mu = weights / w_sum if w_sum > 1e-12 else torch.full((N,), 1.0 / N, device=input.device)

    return _pack_result("IDGI", attr, d_list, df_list, f_vals, gnorms, mu, N, t0)
