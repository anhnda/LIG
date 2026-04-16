
from __future__ import annotations

import math
import time
import torch
import torch.nn as nn

from utility import (
    AttributionResult,
    _forward_scalar,
    _forward_and_gradient,
    _pack_result,
)


EPSILON = 1e-9


def _translate_alpha_to_x(alpha: float, x_input: torch.Tensor,
                          x_baseline: torch.Tensor) -> torch.Tensor:
    """x(alpha) along straight line from baseline to input."""
    return x_baseline + (x_input - x_baseline) * alpha  # x(alpha) = x_b + alpha * (x_i - x_b)


def _translate_x_to_alpha(x: torch.Tensor, x_input: torch.Tensor,
                          x_baseline: torch.Tensor) -> torch.Tensor:
    """
    Per-feature alpha for current x inside [baseline, input] interval.
    Returns NaN where input equals baseline (same as PAIR implementation).
    """
    diff = x_input - x_baseline
    out = torch.full_like(x, float("nan"))
    nonzero = diff != 0
    out[nonzero] = (x[nonzero] - x_baseline[nonzero]) / diff[nonzero]  # alpha_i = (x_i - x_bi)/(x_in_i - x_bi)
    return out


def compute_guided_ig_standard(
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
            - N: Number of integration steps (default: 200)
            - fraction: Fraction of low-|grad| features updated per inner step (default: 0.25)
            - max_dist: Max relative L1 deviation from straight line in alpha-space (default: 0.02)

    Returns:
        AttributionResult containing attributions and diagnostics.
    """
    baseline = params["baseline"]
    N = int(params.get("N", 200))
    fraction = float(params.get("fraction", 0.25))
    max_dist = float(params.get("max_dist", 0.02))

    if N <= 0:
        raise ValueError(f"N must be > 0, got {N}")
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if not (0.0 <= max_dist <= 1.0):
        raise ValueError(f"max_dist must be in [0, 1], got {max_dist}")
    if baseline.shape != input.shape:
        raise ValueError(f"baseline shape {baseline.shape} != input shape {input.shape}")

    t0 = time.time()
    x_input = input
    x_baseline = baseline
    x = x_baseline.clone()
    total_diff = x_input - x_baseline
    l1_total = float(total_diff.abs().sum())  # d_total = ||x_input - x_baseline||_1

    f_bl = _forward_scalar(model, x_baseline)
    f_x = _forward_scalar(model, x_input)
    target = f_x - f_bl

    attr = torch.zeros_like(x_input)
    d_list: list[float] = []
    df_list: list[float] = []
    gnorms: list[float] = []
    f_vals: list[float] = [f_bl]
    gamma_pts: list[torch.Tensor] = [x_baseline.clone()]

    if l1_total <= EPSILON:
        mu = torch.full((N,), 1.0 / N, device=input.device)
        return _pack_result(
            "Guided IG (standard)",
            attr,
            d_list=[0.0] * N,
            df_list=[0.0] * N,
            f_vals=[f_bl] * (N + 1),
            gnorms=[0.0] * N,
            mu=mu,
            N=N,
            t0=t0,
        )

    for step in range(N):
        step_start = x.clone()

        _, grad_actual = _forward_and_gradient(model, x)
        grad = grad_actual.clone()
        gnorms.append(float(grad_actual.norm()))

        alpha = (step + 1.0) / N  # alpha_t = t / T
        alpha_min = max(alpha - max_dist, 0.0)  # lower bound of allowed alpha window
        alpha_max = min(alpha + max_dist, 1.0)  # upper bound of allowed alpha window

        x_min = _translate_alpha_to_x(alpha_min, x_input, x_baseline)  # x(alpha_min)
        x_max = _translate_alpha_to_x(alpha_max, x_input, x_baseline)  # x(alpha_max)

        l1_target = l1_total * (1.0 - (step + 1.0) / N)  # d_target = d_total * (1 - t/T)

        gamma = float("inf")
        while gamma > 1.0:
            x_old = x.clone()

            x_alpha = _translate_x_to_alpha(x, x_input, x_baseline)
            x_alpha = torch.where(torch.isnan(x_alpha),
                                  torch.full_like(x_alpha, alpha_max),
                                  x_alpha)

            # Features behind the allowed interval are pulled up to x_min.
            behind = x_alpha < alpha_min  # features lagging behind alpha window
            x[behind] = x_min[behind]

            l1_current = float((x - x_input).abs().sum())  # d_current = ||x - x_input||_1
            if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
                attr += (x - x_old) * grad_actual
                break

            # Exclude already-saturated-at-x_max features from selection.
            grad[x == x_max] = float("inf")

            abs_grad = grad.abs().reshape(-1)
            threshold = torch.quantile(abs_grad, fraction, interpolation="lower")  # p-quantile of |grad|
            select = (grad.abs() <= threshold) & torch.isfinite(grad)  # S = {i : |g_i| <= threshold}

            l1_s = float(((x - x_max).abs() * select).sum())  # d_S: max reducible L1 using selected features
            if l1_s > 0.0:
                gamma = (l1_current - l1_target) / l1_s  # gamma = (d_current - d_target) / d_S
            else:
                gamma = float("inf")

            if gamma > 1.0:
                x[select] = x_max[select]  # move selected features fully to x_max
            else:
                if gamma <= 0.0:
                    raise RuntimeError(f"Invalid gamma={gamma}")
                x[select] = x[select] + (x_max[select] - x[select]) * gamma  # x <- (1-gamma)x + gamma*x_max

            attr += (x - x_old) * grad_actual  # accumulate integral increment: grad(x_old) * delta_x

        step_delta = x - step_start
        d_list.append(float((grad_actual * step_delta).sum()))  # d_k = <grad_k, Delta gamma_k>

        f_prev = f_vals[-1]
        f_next = _forward_scalar(model, x)
        df_list.append(f_next - f_prev)  # Delta f_k = f(x_{k+1}) - f(x_k)
        f_vals.append(f_next)
        gamma_pts.append(x.clone())

    mu = torch.full((N,), 1.0 / N, device=input.device)
    return _pack_result("Guided IG (standard)", attr, d_list, df_list, f_vals,
                        gnorms, mu, N, t0, gamma_pts=gamma_pts)
