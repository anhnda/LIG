
from __future__ import annotations

import time
import torch
import torch.nn as nn

from utility import (
    AttributionResult,
    _forward_scalar,
    _forward_and_gradient,
    _pack_result,
)


def compute_idgi_standard(
    model: nn.Module,
    input: torch.Tensor,
    params: dict,
) -> AttributionResult:
    """

    Args:
        model: PyTorch model that outputs scalar logits (use ClassLogitModel wrapper)
        input: Input tensor (1, C, H, W)
        params: Dictionary with:
            - baseline: Baseline tensor (1, C, H, W)
            - N: Number of straight-line segments (default: 50)

    Returns:
        AttributionResult containing attributions and diagnostics.
    """
    baseline = params["baseline"]
    N = int(params.get("N", 50))

    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    if baseline.shape != input.shape:
        raise ValueError(f"baseline shape {baseline.shape} != input shape {input.shape}")

    t0 = time.time()
    device = input.device
    dtype = input.dtype

    delta_x = input - baseline
    attr = torch.zeros_like(input)

    # Straight-line path points x_j = x' + (j/N)(x - x'), j=0..N
    alphas = torch.arange(N + 1, device=device, dtype=dtype).view(N + 1, 1, 1, 1) / N
    path = baseline + alphas * delta_x

    d_list: list[float] = []
    df_list: list[float] = []
    gnorms: list[float] = []
    f_vals: list[float] = []

    for j in range(N):
        x_j = path[j:j + 1]
        x_j1 = path[j + 1:j + 2]

        f_j, g_j = _forward_and_gradient(model, x_j)
        f_j1 = _forward_scalar(model, x_j1)

        d_j = f_j1 - f_j
        g_sq = g_j.pow(2)
        g_norm_sq = float(g_sq.sum())

        if g_norm_sq > 1e-12:
            attr += g_sq * (d_j / g_norm_sq)

        d_list.append(float((g_j * (x_j1 - x_j)).sum()))
        df_list.append(float(d_j))
        gnorms.append(float(g_j.norm()))
        f_vals.append(float(f_j))

    f_vals.append(float(_forward_scalar(model, path[N:N + 1])))

    # Diagnostic measure used by IDGI analysis (μ_k ∝ |Δf_k|).
    df_arr = torch.tensor(df_list, device=device)
    w = df_arr.abs()
    w_sum = w.sum()
    mu = w / w_sum if w_sum > 1e-12 else torch.full((N,), 1.0 / N, device=device)

    return _pack_result("IDGI (standard)", attr, d_list, df_list, f_vals, gnorms, mu, N, t0)
