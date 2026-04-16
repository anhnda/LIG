"""
lig_idgi.py - μ-Optimized IG with Signal Harvesting (u-Optimizer)
==================================================================

μ-Optimized IG: optimized measure μ*, straight-line path.
Minimizes: Var_ν(φ) − λ Σ_k μ_k |d_k| + (τ/2) ‖μ‖²₂

This is the signal-harvesting variant that combines conservation with
signal harvesting. Also known as "u-Optimizer" in the paper.

Usage:
    from lig_idgi import compute_lig_idgi

    attribution_result = compute_lig_idgi(
        model=model,
        input=x,
        params={'baseline': baseline, 'N': 50, 'lam': 1.0, 'tau': 0.01}
    )
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn

from utility import (
    AttributionResult,
    _straight_line_pass,
    _pack_result,
    _rescale,
    optimize_mu_signal_harvesting,
)


def compute_lig_idgi(
    model: nn.Module,
    input: torch.Tensor,
    params: dict,
) -> AttributionResult:
    """
    Compute μ-Optimized IG with signal harvesting.

    Args:
        model: PyTorch model that outputs scalar logits (use ClassLogitModel wrapper)
        input: Input tensor (1, C, H, W)
        params: Dictionary with:
            - baseline: Baseline tensor (1, C, H, W)
            - N: Number of interpolation steps (default: 50)
            - lam: Signal-harvesting strength λ (default: 1.0)
            - tau: L2 admissibility multiplier (default: 0.01)
            - n_iter: Number of optimization iterations (default: 300)

    Returns:
        AttributionResult containing attributions and metrics

    Special cases:
        lam=0 : pure conservation (original LAM μ-optimisation)
        lam→∞ : recovers IDGI (μ_k ∝ |d_k|)
    """
    baseline = params['baseline']
    N = params.get('N', 50)
    lam = params.get('lam', 1.0)
    tau = params.get('tau', 0.01)
    n_iter = params.get('n_iter', 300)

    t0 = time.time()

    # Compute gradients along straight line
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, input, baseline, N)

    d_arr = torch.tensor(d_list, device=input.device)
    df_arr = torch.tensor(df_list, device=input.device)

    # Optimize μ with signal-harvesting objective
    mu = optimize_mu_signal_harvesting(
        d_arr, df_arr, lam=lam, tau=tau, n_iter=n_iter)

    # Weighted gradient sum
    grad_stack = torch.cat(grads, dim=0)                 # (N, C, H, W)
    mu_4d = mu.view(N, 1, 1, 1)
    wg = (mu_4d * grad_stack).sum(dim=0, keepdim=True)   # (1, C, H, W)
    attr = _rescale(delta_x * wg, target)

    return _pack_result("LIG-IDIG", attr, d_list, df_list, f_vals,
                        gnorms, mu, N, t0)
