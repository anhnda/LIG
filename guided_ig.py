
from __future__ import annotations

import time
import torch
import torch.nn as nn

from utility import (
    AttributionResult,
    _forward_scalar,
    _forward_and_gradient,
    _rescale,
    _pack_result,
    _dot,
)


def compute_guided_ig(
    model: nn.Module,
    input: torch.Tensor,
    params: dict,
) -> AttributionResult:
    """
    Compute Guided IG attribution.

    Args:
        model: PyTorch model that outputs scalar logits (use ClassLogitModel wrapper)
        input: Input tensor (1, C, H, W)
        params: Dictionary with:
            - baseline: Baseline tensor (1, C, H, W)
            - N: Number of interpolation steps (default: 50)

    Returns:
        AttributionResult containing attributions and metrics

    Note: This method is inherently sequential — each step depends on the
    previous one — so it cannot be batched like straight-line methods.
    """
    baseline = params['baseline']
    N = params.get('N', 50)

    t0 = time.time()
    device = input.device
    delta_x = input - baseline
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, input)
    target = f_x - f_bl

    remaining = delta_x.clone()
    current = baseline.clone()
    gamma_pts = [current.clone()]
    grad_list, d_list, df_list, gnorms = [], [], [], []
    f_vals = [f_bl]

    for k in range(N):
        f_k, grad_k = _forward_and_gradient(model, current)
        grad_list.append(grad_k)
        gnorms.append(float(grad_k.norm()))

        # Inverse-gradient weighting: small |grad| → move more
        abs_g = grad_k.abs() + 1e-8
        inv_w = 1.0 / abs_g
        frac = inv_w / inv_w.sum()
        remaining_steps = N - k

        raw_step = remaining.abs() * frac * remaining_steps * remaining.numel()
        step = remaining.sign() * torch.minimum(raw_step, remaining.abs())

        next_pt = current + step
        f_k1 = _forward_scalar(model, next_pt)

        d_list.append(_dot(grad_k, step))
        df_list.append(f_k1 - f_k)
        f_vals.append(f_k1)

        remaining = remaining - step
        current = next_pt
        gamma_pts.append(current.clone())

    # Attribution: sum of gradient × step at each point
    attr = torch.zeros_like(input)
    for k in range(N):
        attr += grad_list[k] * (gamma_pts[k + 1] - gamma_pts[k])
    attr = _rescale(attr, target)

    mu = torch.full((N,), 1.0 / N, device=device)
    result = _pack_result("Guided IG", attr, d_list, df_list, f_vals,
                          gnorms, mu, N, t0, gamma_pts=gamma_pts)
    return result
