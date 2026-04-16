"""
lig.py - Joint* - Full Signal-Harvesting Solution (LIG)
========================================================

Joint*: Alternating minimization of γ and μ under the signal-harvesting objective.
Minimizes: Var_ν(φ) − λ Σ_k μ_k |d_k| + (τ/2) ‖μ‖²₂

This is the complete signal-harvesting solution that jointly optimizes both
the path γ and the measure μ.

Usage:
    from lig import compute_lig

    attribution_result = compute_lig(
        model=model,
        input=x,
        params={
            'baseline': baseline,
            'N': 50,
            'lam': 1.0,
            'tau': 0.01,
            'G': 16,
            'patch_size': 14,
            'n_alternating': 2,
            'mu_iter': 300,
            'path_iter': 10
        }
    )
"""

from __future__ import annotations

import time
from typing import Optional
import torch
import torch.nn as nn

from utility import (
    AttributionResult,
    _forward_scalar,
    _rescale,
    _pack_result,
    _gradient_batch,
    optimize_mu_signal_harvesting,
    optimize_path_signal_harvesting,
    compute_all_metrics,
    compute_signal_harvesting_objective,
)


def compute_lig(
    model: nn.Module,
    input: torch.Tensor,
    params: dict,
) -> AttributionResult:
    """
    Compute Joint* (LIG) attribution with signal harvesting.

    Args:
        model: PyTorch model that outputs scalar logits (use ClassLogitModel wrapper)
        input: Input tensor (1, C, H, W)
        params: Dictionary with:
            - baseline: Baseline tensor (1, C, H, W)
            - N: Number of interpolation steps (default: 50)
            - lam: Signal-harvesting strength λ (default: 1.0)
            - tau: L2 admissibility multiplier (default: 0.01)
            - G: Number of spatial groups (default: 16)
            - patch_size: Patch size for grouping (default: 14)
            - n_alternating: Number of alternating iterations (default: 2)
            - mu_iter: Iterations for μ optimization (default: 300)
            - path_iter: Iterations for path optimization (default: 10)
            - init_path: Optional initial path as list of N+1 tensors

    Returns:
        AttributionResult containing attributions and metrics
    """
    baseline = params['baseline']
    N = params.get('N', 50)
    lam = params.get('lam', 1.0)
    tau = params.get('tau', 0.01)
    G = params.get('G', 16)
    patch_size = params.get('patch_size', 14)
    n_alternating = params.get('n_alternating', 2)
    mu_iter = params.get('mu_iter', 300)
    path_iter = params.get('path_iter', 10)
    init_path = params.get('init_path', None)

    t0 = time.time()
    device = input.device
    delta_x = input - baseline
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, input)
    target = f_x - f_bl

    # Initialize path
    if init_path is not None:
        assert len(init_path) == N + 1, \
            f"init_path must have N+1={N+1} points, got {len(init_path)}"
        gamma_pts = [p.clone() for p in init_path]
    else:
        gamma_pts = [baseline + (k / N) * delta_x for k in range(N + 1)]

    mu = torch.full((N,), 1.0 / N, device=device)
    Q_history = []

    # Track best state
    best_obj = float("inf")
    best_Q = -1.0
    best_gamma_pts = gamma_pts
    best_mu = mu
    best_d_list: list[float] = []
    best_df_list: list[float] = []
    best_f_vals: list[float] = []
    best_gnorms: list[float] = []
    best_grads: list[torch.Tensor] = []

    def _evaluate_path(gp, mu_vec):
        """Batched evaluation of a path. Returns diagnostics."""
        ap = torch.cat(gp, dim=0)  # (N+1, C, H, W)
        with torch.no_grad():
            fa = model(ap)  # (N+1,)
        pn = ap[:N]
        gb = _gradient_batch(model, pn)  # (N, C, H, W)
        sb = ap[1:] - ap[:N]  # steps
        dt = (gb * sb).view(N, -1).sum(dim=1)  # d_k
        dl = dt.tolist()

        f0 = fa[0].item()
        fv = [f0] + fa.tolist()  # N+2 entries
        dfl = [fv[k + 1] - fv[k] for k in range(N)]

        gn = gb.view(N, -1).norm(dim=1).tolist()
        gr = [gb[k:k+1].clone() for k in range(N)]
        da = torch.tensor(dl, device=device)
        dfa = torch.tensor(dfl, device=device)
        return dl, dfl, fv, gn, gr, da, dfa

    for s in range(n_alternating):
        # Evaluate current path
        d_list, df_list, f_vals, gnorms, grads, d_arr, df_arr = \
            _evaluate_path(gamma_pts, mu)

        # Phase 1: optimize μ
        mu = optimize_mu_signal_harvesting(
            d_arr, df_arr, lam=lam, tau=tau, n_iter=mu_iter)

        var_mu, cv2_mu, Q_mu = compute_all_metrics(d_arr, df_arr, mu)
        obj_mu, _, _, _ = compute_signal_harvesting_objective(
            d_arr, df_arr, mu, lam=lam, tau=tau)

        # Update best if improved
        if obj_mu < best_obj or (abs(obj_mu - best_obj) < 1e-8 and Q_mu > best_Q):
            best_obj = obj_mu
            best_Q = Q_mu
            best_gamma_pts = gamma_pts
            best_mu = mu.clone()
            best_d_list, best_df_list = d_list, df_list
            best_f_vals, best_gnorms, best_grads = f_vals, gnorms, grads

        # Phase 2: optimize path
        Q_path = Q_mu
        obj_path = obj_mu
        if s < n_alternating - 1:
            new_gamma_pts = optimize_path_signal_harvesting(
                model, input, baseline, mu, N=N, G=G,
                patch_size=patch_size, n_iter=path_iter, lr=0.08,
                lam=lam)

            # Evaluate the new path
            new_d_list, new_df_list, new_f_vals, new_gnorms, new_grads, \
                new_d_arr, new_df_arr = _evaluate_path(new_gamma_pts, mu)
            _, _, Q_new = compute_all_metrics(new_d_arr, new_df_arr, mu)
            obj_new, _, _, _ = compute_signal_harvesting_objective(
                new_d_arr, new_df_arr, mu, lam=lam, tau=tau)

            Q_path = Q_new
            obj_path = obj_new

            # Regression guard: only accept if objective improved
            if obj_new < best_obj:
                gamma_pts = new_gamma_pts
                d_list, df_list = new_d_list, new_df_list
                f_vals, gnorms, grads = new_f_vals, new_gnorms, new_grads
                d_arr, df_arr = new_d_arr, new_df_arr

                best_obj = obj_new
                best_Q = Q_new
                best_gamma_pts = new_gamma_pts
                best_mu = mu.clone()
                best_d_list, best_df_list = new_d_list, new_df_list
                best_f_vals, best_gnorms, best_grads = new_f_vals, new_gnorms, new_grads
            else:
                # Revert to best known state
                gamma_pts = best_gamma_pts
                mu = best_mu.clone()

        Q_history.append({
            "iteration": s,
            "Q_after_mu": float(Q_mu),
            "Q_after_path": float(Q_path),
            "obj_after_mu": float(obj_mu),
            "obj_after_path": float(obj_path),
            "best_Q": float(best_Q),
            "best_obj": float(best_obj),
        })

    # Use best state for final attributions
    gamma_pts = best_gamma_pts
    mu = best_mu
    grads = best_grads

    attr = torch.zeros_like(input)
    for k in range(N):
        attr += mu[k] * grads[k] * (gamma_pts[k + 1] - gamma_pts[k])
    attr = _rescale(attr, target)

    return _pack_result("LIG", attr, best_d_list, best_df_list,
                        best_f_vals, best_gnorms, mu, N, t0, Q_history,
                        gamma_pts=gamma_pts)
