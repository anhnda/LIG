"""
unified_ig.py — Unified Integrated Gradients Framework (v3)
=============================================================

All IG variants minimise the same objective — the Snell's law conservation
violation:

    min_{γ,μ}  Var_ν(φ)  =  Σ_k ν_k (φ_k − φ̄_ν)²

where:
    φ_k = d_k / Δf_k         step fidelity
    ν_k = μ_k Δf_k² / Σ …   effective measure

Derived quantities reported alongside:
    CV²(φ)  = Var_ν(φ) / φ̄²         (scale-free)
    𝒬       = 1 / (1 + CV²(φ))      (quality score, 1 = perfect)

Methods and what they optimise:
    IG           — nothing           (straight line, uniform μ)
    IDGI         — μ heuristic       (straight line, μ_k ∝ |Δf_k|)
    Guided IG    — γ heuristic       (low-grad-first path, uniform μ)
    μ-Optimised  — μ optimal         (straight line, min Var_ν(φ))
    Joint        — γ + μ optimal     (alternating minimisation)

Efficiency notes:
    • Fused _forward_and_gradient → 1 backward pass per step (not 2)
    • φ, Δf² hoisted out of Adam loop in optimize_mu
    • Spatial group cache in joint path optimisation
    • Batched insertion/deletion evaluation

Usage:
    python unified_ig.py                        # single run
    python unified_ig.py --viz --viz-fidelity   # with plots
    python unified_ig.py --json results.json    # export

Requirements: torch >= 2.0, torchvision
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T

from utils import (
    get_device, AttributionResult, StepInfo, InsDelScores,
    compute_Var_nu, compute_CV2, compute_Q, compute_all_metrics,
    compute_insertion_deletion, run_insertion_deletion,
    visualize_step_fidelity, visualize_insertion_deletion,
)


# ═════════════════════════════════════════════════════════════════════════════
# §1  MODEL WRAPPER
# ═════════════════════════════════════════════════════════════════════════════

class ClassLogitModel(nn.Module):
    """Wrap a classifier → scalar logit for a target class.  Shape: (B,)."""

    def __init__(self, backbone: nn.Module, target_class: int):
        super().__init__()
        self.backbone = backbone
        self.target_class = target_class

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)[:, self.target_class]


# ═════════════════════════════════════════════════════════════════════════════
# §2  GRADIENT UTILITIES  (fused forward + backward)
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _forward_scalar(model: nn.Module, x: torch.Tensor) -> float:
    return float(model(x).squeeze())


def _forward_and_gradient(model: nn.Module, x: torch.Tensor
                          ) -> tuple[float, torch.Tensor]:
    """f(x) and ∇f(x) in ONE backward pass."""
    with torch.enable_grad():
        x_in = x.detach().clone().requires_grad_(True)
        model.zero_grad()
        out = model(x_in).sum()
        f_val = float(out)
        out.backward()
    return f_val, x_in.grad.detach()


def _gradient(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """∇f(x) only (when f value already known)."""
    with torch.enable_grad():
        x_in = x.detach().clone().requires_grad_(True)
        model.zero_grad()
        model(x_in).sum().backward()
    return x_in.grad.detach()


def _dot(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a * b).sum())


# ═════════════════════════════════════════════════════════════════════════════
# §3  STEP DIAGNOSTICS BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def _rescale(attr: torch.Tensor, target: float) -> torch.Tensor:
    s = attr.sum().item()
    return attr * (target / s) if abs(s) > 1e-12 else attr


def _build_steps(d_list, df_list, f_vals, gnorms, mu, N) -> list[StepInfo]:
    steps = []
    for k in range(N):
        dk, dfk = d_list[k], df_list[k]
        rk = dfk - dk
        phik = dk / dfk if abs(dfk) > 1e-12 else 1.0
        steps.append(StepInfo(
            t=k / N, f=f_vals[k], d_k=dk, delta_f_k=dfk,
            r_k=rk, phi_k=phik, grad_norm=gnorms[k], mu_k=float(mu[k]),
        ))
    return steps


def _pack_result(name, attr, d_list, df_list, f_vals, gnorms, mu, N,
                 t0, Q_history=None) -> AttributionResult:
    """Build AttributionResult with all three metrics in one pass."""
    device = attr.device
    d_arr = torch.tensor(d_list, device=device)
    df_arr = torch.tensor(df_list, device=device)
    var_nu, cv2, Q = compute_all_metrics(d_arr, df_arr, mu)
    steps = _build_steps(d_list, df_list, f_vals, gnorms, mu, N)
    return AttributionResult(
        name=name, attributions=attr, Q=Q, CV2=cv2, Var_nu=var_nu,
        steps=steps, Q_history=Q_history or [], elapsed_s=time.time() - t0,
    )


# ═════════════════════════════════════════════════════════════════════════════
# §4  STRAIGHT-LINE PASS  (shared by IG, IDGI, μ-Optimised)
#
#     Cost: 2 forward + N fused(forward+backward) = N+2 model evaluations
# ═════════════════════════════════════════════════════════════════════════════

def _straight_line_pass(model: nn.Module, x: torch.Tensor,
                        baseline: torch.Tensor, N: int):
    """
    Evaluate f and ∇f at N uniformly-spaced points along the straight line.

    Returns: (delta_x, target, grads, d_list, df_list, f_vals, gnorms)
        grads   : list of N gradient tensors
        d_list  : list of N floats (d_k = ∇f·Δγ_k)
        df_list : list of N floats (Δf_k)
        f_vals  : list of N+1 floats (f at each γ point)
        gnorms  : list of N floats (‖∇f‖)
    """
    delta_x = x - baseline
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, x)
    target = f_x - f_bl
    step = delta_x / N

    grads, d_list, gnorms = [], [], []
    f_vals = [f_bl]

    for k in range(N):
        gamma_k = baseline + (k / N) * delta_x
        f_k, grad_k = _forward_and_gradient(model, gamma_k)
        f_vals.append(f_k)
        grads.append(grad_k)
        d_list.append(_dot(grad_k, step))
        gnorms.append(float(grad_k.norm()))

    f_vals.append(f_x)  # reuse — no extra forward
    df_list = [f_vals[k + 1] - f_vals[k] for k in range(N)]

    return delta_x, target, grads, d_list, df_list, f_vals, gnorms


# ═════════════════════════════════════════════════════════════════════════════
# §5  STANDARD IG
# ═════════════════════════════════════════════════════════════════════════════

def standard_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
                N: int = 50) -> AttributionResult:
    """Standard IG (Sundararajan et al., 2017).  No optimisation."""
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    grad_sum = sum(grads)
    attr = delta_x * grad_sum / N
    mu = torch.full((N,), 1.0 / N, device=x.device)

    return _pack_result("IG", attr, d_list, df_list, f_vals, gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §6  IDGI
# ═════════════════════════════════════════════════════════════════════════════

def idgi(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
         N: int = 50) -> AttributionResult:
    """IDGI (Sikdar et al., 2021).  μ_k ∝ |Δf_k| on straight line."""
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    df_arr = torch.tensor(df_list, device=x.device)
    weights = df_arr.abs()
    w_sum = weights.sum()
    mu = weights / w_sum if w_sum > 1e-12 else torch.full((N,), 1.0/N, device=x.device)

    wg = sum(mu[k].item() * grads[k] for k in range(N))
    attr = _rescale(delta_x * wg, target)

    return _pack_result("IDGI", attr, d_list, df_list, f_vals, gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §7  GUIDED IG
# ═════════════════════════════════════════════════════════════════════════════

def guided_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
              N: int = 50) -> AttributionResult:
    """Guided IG (Kapishnikov et al., 2021).  Low-grad-first path, uniform μ."""
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, x)
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

    # Attribution
    attr = torch.zeros_like(x)
    for k in range(N):
        attr += grad_list[k] * (gamma_pts[k + 1] - gamma_pts[k])
    attr = _rescale(attr, target)

    mu = torch.full((N,), 1.0 / N, device=device)
    return _pack_result("Guided IG", attr, d_list, df_list, f_vals,
                        gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §8  μ-OPTIMISATION  (Phase 1)
#
#     Objective aligned with paper:  min  Var_ν(φ) + τ·H(μ)
#     φ and Δf² are constants → hoisted out of Adam loop
# ═════════════════════════════════════════════════════════════════════════════

def optimize_mu(d: torch.Tensor, delta_f: torch.Tensor,
                tau: float = 0.01, n_iter: int = 200,
                lr: float = 0.05) -> torch.Tensor:
    """
    Find μ minimising  Var_ν(φ) + τ·Σ μ_k log μ_k.

    This directly minimises the paper's primary objective (Var, not CV²).
    Adam on softmax logits → unconstrained optimisation on the simplex.
    """
    device = d.device
    N = d.shape[0]

    # Constants w.r.t. μ — hoist out of loop
    valid = delta_f.abs() > 1e-12
    safe = torch.where(valid, delta_f, torch.ones_like(delta_f))
    phi = torch.where(valid, d / safe, torch.ones_like(d)).detach()
    df2 = (delta_f ** 2).detach()

    logits = torch.zeros(N, device=device, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=lr)

    for _ in range(n_iter):
        opt.zero_grad()
        mu = torch.softmax(logits, dim=0)

        # Effective measure ν_k = μ_k Δf_k², normalised
        nu = mu * df2
        nu_sum = nu.sum()
        if nu_sum < 1e-15:
            break
        w = nu / nu_sum

        mean_phi = (w * phi).sum()
        var_phi = (w * (phi - mean_phi) ** 2).sum()

        # OBJECTIVE: Var_ν(φ), NOT CV²(φ)
        # This directly minimises the conservation law violation
        entropy = (mu * torch.log(mu + 1e-15)).sum()
        loss = var_phi + tau * entropy

        loss.backward()
        opt.step()

    with torch.no_grad():
        mu = torch.softmax(logits, dim=0)
    return mu.detach()


# ═════════════════════════════════════════════════════════════════════════════
# §9  μ-OPTIMISED IG
# ═════════════════════════════════════════════════════════════════════════════

def mu_optimized_ig(model: nn.Module, x: torch.Tensor,
                    baseline: torch.Tensor, N: int = 50,
                    tau: float = 0.005, n_iter: int = 300
                    ) -> AttributionResult:
    """Straight line + optimal μ.  Cost = standard IG + O(N) arithmetic."""
    t0 = time.time()
    delta_x, target, grads, d_list, df_list, f_vals, gnorms = \
        _straight_line_pass(model, x, baseline, N)

    d_arr = torch.tensor(d_list, device=x.device)
    df_arr = torch.tensor(df_list, device=x.device)
    mu = optimize_mu(d_arr, df_arr, tau=tau, n_iter=n_iter)

    wg = sum(mu[k].item() * grads[k] for k in range(N))
    attr = _rescale(delta_x * wg, target)

    return _pack_result("μ-Optimized", attr, d_list, df_list, f_vals,
                        gnorms, mu, N, t0)


# ═════════════════════════════════════════════════════════════════════════════
# §10  PATH OPTIMISATION  (Phase 2)
#
#     Spatial grouping for images + softmax velocity scheduling.
#     Objective: Var_ν(φ), consistent with §8.
# ═════════════════════════════════════════════════════════════════════════════

_group_cache: dict = {}   # memoised spatial groups


def _build_spatial_groups(model, x, baseline, G=16, patch_size=14):
    """Assign pixels to G groups by gradient importance.  Cached."""
    key = (x.data_ptr(), baseline.data_ptr(), G, patch_size)
    if key in _group_cache:
        return _group_cache[key]

    device = x.device
    _, C, H, W = x.shape
    delta_x = x - baseline
    mid = baseline + 0.5 * delta_x
    grad_mid = _gradient(model, mid)
    importance = (grad_mid * delta_x).abs().sum(dim=1, keepdim=True)

    n_rows = (H + patch_size - 1) // patch_size
    n_cols = (W + patch_size - 1) // patch_size
    n_patches = n_rows * n_cols

    patch_imp = torch.zeros(n_patches, device=device)
    patch_map = torch.zeros(1, 1, H, W, dtype=torch.long, device=device)

    for r in range(n_rows):
        for c in range(n_cols):
            pid = r * n_cols + c
            r0, r1 = r * patch_size, min((r + 1) * patch_size, H)
            c0, c1 = c * patch_size, min((c + 1) * patch_size, W)
            patch_map[0, 0, r0:r1, c0:c1] = pid
            patch_imp[pid] = importance[0, 0, r0:r1, c0:c1].mean()

    order = torch.argsort(patch_imp)
    p2g = torch.zeros(n_patches, dtype=torch.long, device=device)
    per_grp = n_patches // G
    for g in range(G):
        lo = g * per_grp
        hi = (g + 1) * per_grp if g < G - 1 else n_patches
        p2g[order[lo:hi]] = g

    gmap = p2g[patch_map.flatten()].view(1, 1, H, W)
    _group_cache[key] = gmap
    return gmap


def _build_path_2d(baseline, delta_x, V, group_map, N):
    """Path from grouped velocity schedule V (G, N)."""
    G = V.shape[0]
    gamma = [baseline.clone()]
    v_sums = V.sum(dim=1, keepdim=True).clamp(min=1e-12)
    for k in range(N):
        step = torch.zeros_like(baseline)
        for g in range(G):
            mask = (group_map == g).expand_as(baseline)
            step[mask] = delta_x[mask] * (V[g, k] / v_sums[g, 0])
        gamma.append(gamma[-1] + step)
    return gamma


def optimize_path(model, x, baseline, mu, N=50, G=16, patch_size=14,
                  n_iter=15, lr=0.05):
    """
    Optimise path via grouped velocity scheduling.

    Objective: Var_ν(φ), matching optimize_mu.
    Softmax over time dim per group → normalisation built in.
    """
    device = x.device
    delta_x = x - baseline
    gmap = _build_spatial_groups(model, x, baseline, G, patch_size)

    log_V = torch.zeros(G, N, device=device)
    best_var, best_logV = float("inf"), log_V.clone()

    def _var_of(lv):
        V = torch.softmax(lv, dim=1) * N
        gp = _build_path_2d(baseline, delta_x, V, gmap, N)
        d_v = torch.zeros(N, device=device)
        df_v = torch.zeros(N, device=device)
        f_prev = _forward_scalar(model, gp[0])
        for kk in range(N):
            grd = _gradient(model, gp[kk])
            d_v[kk] = _dot(grd, gp[kk + 1] - gp[kk])
            f_next = _forward_scalar(model, gp[kk + 1])
            df_v[kk] = f_next - f_prev
            f_prev = f_next
        return compute_Var_nu(d_v, df_v, mu)     # ← Var_ν, NOT CV²

    eps = 0.1
    for it in range(n_iter):
        cur_var = _var_of(log_V)
        if cur_var < best_var:
            best_var, best_logV = cur_var, log_V.clone()

        # Probe one group per iteration (cycle), all N time steps
        g = it % G
        grad_lv = torch.zeros_like(log_V)
        for k in range(N):
            log_V[g, k] += eps
            grad_lv[g, k] = (_var_of(log_V) - cur_var) / eps
            log_V[g, k] -= eps

        log_V = log_V - lr * grad_lv

    best_V = torch.softmax(best_logV, dim=1) * N
    return _build_path_2d(baseline, delta_x, best_V, gmap, N)


# ═════════════════════════════════════════════════════════════════════════════
# §11  JOINT OPTIMISATION  (alternating γ + μ)
# ═════════════════════════════════════════════════════════════════════════════

def joint_ig(model: nn.Module, x: torch.Tensor, baseline: torch.Tensor,
             N: int = 50, G: int = 16, patch_size: int = 14,
             n_alternating: int = 2, tau: float = 0.005,
             mu_iter: int = 300, path_iter: int = 10,
             ) -> AttributionResult:
    """
    Joint optimisation: alternating min of Var_ν(φ) over (γ, μ).

    Each alternating step decreases Var_ν (monotone convergence).
    """
    t0 = time.time()
    device = x.device
    delta_x = x - baseline
    f_bl = _forward_scalar(model, baseline)
    f_x = _forward_scalar(model, x)
    target = f_x - f_bl

    # Initialise: straight line, uniform μ
    gamma_pts = [baseline + (k / N) * delta_x for k in range(N + 1)]
    mu = torch.full((N,), 1.0 / N, device=device)
    Q_history = []
    grads = []

    for s in range(n_alternating):
        # Evaluate current path
        f_vals = [_forward_scalar(model, gamma_pts[k]) for k in range(N + 1)]
        d_list, df_list, gnorms = [], [], []
        grads = []

        for k in range(N):
            grad_k = _gradient(model, gamma_pts[k])
            grads.append(grad_k)
            step_k = gamma_pts[k + 1] - gamma_pts[k]
            d_list.append(_dot(grad_k, step_k))
            df_list.append(f_vals[k + 1] - f_vals[k])
            gnorms.append(float(grad_k.norm()))

        d_arr = torch.tensor(d_list, device=device)
        df_arr = torch.tensor(df_list, device=device)

        # Phase 1: optimise μ  (Var_ν objective)
        mu = optimize_mu(d_arr, df_arr, tau=tau, n_iter=mu_iter)
        var_mu, cv2_mu, Q_mu = compute_all_metrics(d_arr, df_arr, mu)

        # Phase 2: optimise path (skip on last iteration)
        if s < n_alternating - 1:
            gamma_pts = optimize_path(
                model, x, baseline, mu, N=N, G=G,
                patch_size=patch_size, n_iter=path_iter)

            # Re-evaluate on new path
            f_vals = [_forward_scalar(model, gamma_pts[k]) for k in range(N+1)]
            d_list, df_list, gnorms, grads = [], [], [], []
            for k in range(N):
                grad_k = _gradient(model, gamma_pts[k])
                grads.append(grad_k)
                step_k = gamma_pts[k + 1] - gamma_pts[k]
                d_list.append(_dot(grad_k, step_k))
                df_list.append(f_vals[k + 1] - f_vals[k])
                gnorms.append(float(grad_k.norm()))
            d_arr = torch.tensor(d_list, device=device)
            df_arr = torch.tensor(df_list, device=device)
            var_path, cv2_path, Q_path = compute_all_metrics(d_arr, df_arr, mu)
        else:
            var_path, Q_path = var_mu, Q_mu

        Q_history.append({
            "iteration": s,
            "Var_after_mu": float(var_mu), "Q_after_mu": float(Q_mu),
            "Var_after_path": float(var_path), "Q_after_path": float(Q_path),
        })

    # Final attributions
    attr = torch.zeros_like(x)
    for k in range(N):
        attr += mu[k] * grads[k] * (gamma_pts[k + 1] - gamma_pts[k])
    attr = _rescale(attr, target)

    return _pack_result("Joint", attr, d_list, df_list, f_vals,
                        gnorms, mu, N, t0, Q_history)


# ═════════════════════════════════════════════════════════════════════════════
# §12  IMAGE LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_image_and_model(device: torch.device, min_conf: float = 0.70):
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    tf = T.Compose([
        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    x, pc, cf = None, None, None
    source = "none"

    # Try local images
    for sample_dir in ["./sample_imagenet1k", "../sample_imagenet1k",
                       os.path.expanduser("~/sample_imagenet1k")]:
        if not os.path.isdir(sample_dir):
            continue
        try:
            from PIL import Image
            import random
            jpegs = sorted([f for f in os.listdir(sample_dir)
                            if f.lower().endswith(('.jpeg', '.jpg', '.png'))])
            random.shuffle(jpegs)
            print(f"Found {sample_dir} ({len(jpegs)} images)")
            for fname in jpegs:
                try:
                    img = Image.open(os.path.join(sample_dir, fname)).convert("RGB")
                except Exception:
                    continue
                xc = tf(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = F.softmax(backbone(xc), dim=-1)
                    c, pr = p[0].max(0)
                if c.item() >= min_conf:
                    x, pc, cf = xc, pr.item(), c.item()
                    source = f"{sample_dir}/{fname}"
                    print(f"  ✓ {fname} → class={pc}, conf={cf:.4f}")
                    break
        except Exception as e:
            print(f"  Error: {e}")
        if x is not None:
            break

    # Fallback: CIFAR-10
    if x is None:
        try:
            from torchvision.datasets import CIFAR10
            ctf = T.Compose([T.Resize(224), T.ToTensor(),
                             T.Normalize([0.485,0.456,0.406],
                                         [0.229,0.224,0.225])])
            ds = CIFAR10("./data", train=False, download=True, transform=ctf)
            for i in range(500):
                im, _ = ds[i]
                xc = im.unsqueeze(0).to(device)
                with torch.no_grad():
                    p = F.softmax(backbone(xc), dim=-1)
                    c, pr = p[0].max(0)
                if c.item() >= min_conf:
                    x, pc, cf = xc, pr.item(), c.item()
                    source = f"CIFAR-10 idx={i}"
                    break
        except Exception:
            pass

    # Fallback: synthetic
    if x is None:
        print("Using synthetic image fallback")
        m = torch.tensor([0.485,0.456,0.406], device=device).view(1,3,1,1)
        s = torch.tensor([0.229,0.224,0.225], device=device).view(1,3,1,1)
        torch.manual_seed(42)
        raw = (torch.randn(1,3,224,224, device=device)*0.2+0.5).clamp(0,1)
        x = (raw - m) / s
        with torch.no_grad():
            p = F.softmax(backbone(x), dim=-1)
            c, pr = p[0].max(0)
            pc, cf = pr.item(), c.item()
        source = "synthetic"

    model = ClassLogitModel(backbone, target_class=pc).to(device).eval()
    baseline = torch.zeros_like(x)
    info = {"source": source, "target_class": pc, "confidence": cf,
            "model": "ResNet-50 (ImageNet pretrained)"}
    return model, x, baseline, info


# ═════════════════════════════════════════════════════════════════════════════
# §13  HEATMAP VISUALISATION
# ═════════════════════════════════════════════════════════════════════════════

def visualize_attributions(x, methods, info, save_path="attribution_heatmaps.png",
                           delta_f=0.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np

    mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1).to(x.device)
    std = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1).to(x.device)
    img = ((x*std+mean).clamp(0,1)[0].permute(1,2,0).cpu().numpy()*255).astype("uint8")
    img_dark = (img.astype(float)*0.4).astype("uint8")

    cmap = LinearSegmentedColormap.from_list("heat", [
        (0,(0,0,0,0)), (0.3,(0.97,0.45,0.02,0.4)),
        (0.6,(0.97,0.71,0.22,0.7)), (1,(1,1,1,1))])
    colors = {"IG":"#6B7280","IDGI":"#8B5CF6","Guided IG":"#06B6D4",
              "μ-Optimized":"#F59E0B","Joint":"#EF4444"}

    n = len(methods)
    fig, axes = plt.subplots(2, n+1, figsize=(3.6*(n+1), 7.5), facecolor="#0D0D0D")

    axes[0,0].imshow(img); axes[0,0].set_title("Original", color="#E8E4DF",
        fontsize=10, fontfamily="monospace"); axes[0,0].axis("off")

    for i, m in enumerate(methods):
        sal = m.attributions[0].abs().sum(0).cpu().numpy()
        vmax = max(np.percentile(sal, 99), 1e-12)
        sal = (sal/vmax).clip(0,1)
        ax = axes[0, i+1]
        ax.imshow(img_dark); ax.imshow(sal, cmap=cmap, vmin=0, vmax=1, alpha=0.85)
        c = colors.get(m.name, "#F7B538")
        ax.set_title(f"{m.name}\n𝒬={m.Q:.4f}  Var={m.Var_nu:.4f}",
                     color=c, fontsize=9, fontfamily="monospace", linespacing=1.4)
        ax.axis("off")

    # Bottom row: Q bar chart + signed heatmaps
    ax_bar = axes[1, 0]; ax_bar.set_facecolor("#0D0D0D")
    qs = [m.Q for m in methods]; names = [m.name for m in methods]
    cs = [colors.get(n, "#F7B538") for n in names]
    bars = ax_bar.barh(range(n), qs, color=cs, height=0.6)
    for bar, q in zip(bars, qs):
        ax_bar.text(bar.get_width()+0.02, bar.get_y()+bar.get_height()/2,
                    f"{q:.4f}", va="center", color="#E8E4DF", fontsize=8,
                    fontfamily="monospace")
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels(names, fontsize=8, fontfamily="monospace", color="#E8E4DF")
    ax_bar.set_xlim(0, 1.15); ax_bar.invert_yaxis()
    ax_bar.set_title("𝒬 Score", color="#E8E4DF", fontsize=10, fontfamily="monospace")
    ax_bar.tick_params(colors="#888", labelsize=7)
    for sp in ax_bar.spines.values(): sp.set_color("#333")

    cmap_div = LinearSegmentedColormap.from_list("div", [
        (0,(0.15,0.35,0.85,0.9)), (0.5,(0,0,0,0)), (1,(0.95,0.2,0.1,0.9))])
    for i, m in enumerate(methods):
        sal = m.attributions[0].sum(0).cpu().numpy()
        vmax = max(np.percentile(np.abs(sal), 99), 1e-12)
        ax = axes[1, i+1]; ax.imshow(img_dark)
        ax.imshow((sal/vmax).clip(-1,1), cmap=cmap_div, vmin=-1, vmax=1, alpha=0.85)
        ax.set_title(f"Signed · {m.name}", color=colors.get(m.name,"#F7B538"),
                     fontsize=9, fontfamily="monospace"); ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, facecolor="#0D0D0D",
                bbox_inches="tight", pad_inches=0.15)
    plt.close()
    print(f"✓ Heatmap → {save_path}")


# ═════════════════════════════════════════════════════════════════════════════
# §14  EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_experiment(N=50, device=None, min_conf=0.70):
    if device is None:
        device = get_device()

    print("Loading ResNet-50 and image...")
    model, x, baseline, info = load_image_and_model(device, min_conf)

    f_x = _forward_scalar(model, x)
    f_bl = _forward_scalar(model, baseline)
    delta_f = f_x - f_bl

    print(f"\nModel : {info['model']}")
    print(f"Source: {info['source']}")
    print(f"Class : {info['target_class']} (conf={info['confidence']:.4f})")
    print(f"f(x) = {f_x:.4f},  f(bl) = {f_bl:.4f},  Δf = {delta_f:.4f}")
    print(f"N = {N}\n")
    print(f"{'Method':<16} {'Var_ν':>10} {'CV²':>8} {'𝒬':>8} {'Time':>8}")
    print("─" * 56)

    G = 16
    methods = [
        standard_ig(model, x, baseline, N),
        idgi(model, x, baseline, N),
        guided_ig(model, x, baseline, N),
        mu_optimized_ig(model, x, baseline, N, tau=0.005, n_iter=300),
        joint_ig(model, x, baseline, N, G=G, n_alternating=2,
                 tau=0.005, mu_iter=300, path_iter=G),
    ]

    for m in methods:
        print(f"{m.name:<16} {m.Var_nu:>10.6f} {m.CV2:>8.4f} "
              f"{m.Q:>8.4f} {m.elapsed_s:>7.1f}s")

    results = {
        "image_info": info,
        "model_info": {"f_x": f_x, "f_baseline": f_bl,
                       "delta_f": delta_f, "N": N, "device": str(device)},
        "methods": {m.name: m.to_dict() for m in methods},
    }
    return results, methods, model, x, baseline, info


# ═════════════════════════════════════════════════════════════════════════════
# §15  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified IG v3 (PyTorch)")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-conf", type=float, default=0.70)
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--viz-path", type=str, default="attribution_heatmaps.png")
    parser.add_argument("--viz-fidelity", action="store_true")
    parser.add_argument("--insdel", action="store_true")
    parser.add_argument("--insdel-steps", type=int, default=100)
    parser.add_argument("--viz-insdel", action="store_true")
    args = parser.parse_args()

    device = get_device(force=args.device)
    results, methods, model, x, baseline, info = run_experiment(
        N=args.steps, device=device, min_conf=args.min_conf)

    if args.insdel or args.viz_insdel:
        run_insertion_deletion(model, x, baseline, methods,
                               n_steps=args.insdel_steps)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults → {args.json}")

    if args.viz:
        visualize_attributions(x, methods, info, save_path=args.viz_path,
                               delta_f=results["model_info"]["delta_f"])

    if args.viz_fidelity:
        fpath = args.viz_path.replace(".png", "_fidelity.png")
        visualize_step_fidelity(methods, save_path=fpath)

    if args.viz_insdel:
        ipath = args.viz_path.replace(".png", "_insdel.png")
        visualize_insertion_deletion(methods, save_path=ipath)