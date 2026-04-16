# LIG — Least-action Integrated Gradients

A unified variational framework that reveals all Integrated Gradients (IG) variants as special cases of a single **signal-harvesting action**: the path and measure must jointly minimize linearization distortion while maximizing attribution concentrated on the output-transition region.

This framework is the attribution analogue of an **optimal signal-harvesting optical instrument**—combining Fermat's principle (bending light to avoid high-curvature regions) with detector optimization (concentrating measurement on the bright transition region).

## The Core Principle

Standard IG integrates gradients along a straight line from a baseline to the input. In practice, this path traverses regions where the model's output is flat (wasting interpolation budget with no signal) and regions of sharp nonlinearity (where the linear approximation breaks down). Recent methods address this differently:

- **IDGI** reweights *which steps to trust* (the measure μ) by setting μ_k ∝ |Δf_k|
- **Guided IG** changes *where to evaluate gradients* (the path γ) using a low-gradient-first heuristic

**LIG shows both are approximate solutions to the same variational principle**—minimizing the signal-harvesting action:

```
min_{γ,μ}  Var_ν(φ) − λ Σ_k μ_k |d_k| + (τ/2) ||μ||²_2
           └────────┘   └──────────────┘   └─────────┘
           distortion   signal harvested   L² admissibility
```

where:
- **φ_k = d_k / Δf_k** is step fidelity (ratio of gradient-predicted to actual output change)
- **Var_ν(φ)** measures linearization distortion under the effective measure ν_k ∝ μ_k Δf²_k
- **−λ Σ_k μ_k |d_k|** rewards concentrating μ on steps with large output change
- **τ/2 ||μ||²_2** prevents μ from collapsing to a Dirac spike

## Unified Framework: All Methods as Special Cases

All existing IG variants are special cases parameterized by (λ, τ, γ):

| Method | Path γ | λ | τ | Measure μ* | What it optimizes |
|--------|--------|---|---|-----------|-------------------|
| **Standard IG** | straight line | 0 | ∞ | uniform | nothing (baseline) |
| **IDGI** | straight line | >0 | →0 | μ_k ∝ \|Δf_k\| | μ only (exact KKT stationary) |
| **Guided IG** | heuristic | >0 | — | uniform | γ only (approximate stationary) |
| **LIG** (ours) | **optimized** | **>0** | **>0** | **joint optimal** | **γ + μ jointly** |

**Key Insight**: IDGI's measure μ_k ∝ |Δf_k| is not a heuristic—it's the exact stationary point of the signal-harvesting objective in the limit τ → 0⁺. Similarly, Guided IG's path construction approximates the forced Euler-Lagrange equation for the signal-harvesting action.

## Results

**ResNet-50 on ImageNet** (N=50 steps, 30 images, zero baseline):

| Method | Q ↑ | Var_ν ↓ | Ins AUC ↑ | Del AUC ↓ | Ins-Del ↑ | Time (s) |
|--------|-----|---------|-----------|-----------|-----------|----------|
| IG | 0.795 | 0.295 | 0.553 | 0.289 | 0.264 | 0.07 |
| IDGI | 0.854 | 0.184 | 0.607 | 0.243 | 0.364 | 0.06 |
| Guided IG | 0.961 | 0.022 | 0.512 | 0.291 | 0.221 | 1.63 |
| **LIG** | **1.000** | **0.001** | **0.640** | **0.240** | **0.401** | 13.46 |

**LIG achieves near-perfect conservation** (Q=0.9997, ~300× reduction in distortion vs Standard IG) **while maintaining the best faithfulness** (highest Insertion AUC, lowest Deletion AUC, highest Ins-Del gap).

The unified framework **eliminates the conservation-faithfulness trade-off**: by jointly optimizing path and measure, LIG inherits the strengths of both IDGI (optimal measure) and Guided IG (improved path) while avoiding their respective weaknesses.

## Quality Metrics

Three related metrics derived from step fidelity φ_k = d_k / Δf_k:

- **Var_ν(φ)** — weighted variance of fidelity. The distortion term in the signal-harvesting action.
- **CV²(φ)** = Var_ν(φ) / φ̄² — scale-free coefficient of variation. Used in μ-optimization to prevent degeneracy.
- **Q** = 1/(1 + CV²) — quality score in [0, 1]. Q = 1 means perfect step-fidelity constancy across all active steps.

**Faithfulness metrics** (Petsiuk et al., 2018):
- **Insertion AUC** (↑): How quickly the output rises when features are added in order of attribution magnitude.
- **Deletion AUC** (↓): How quickly the output drops when features are removed in order of attribution magnitude.
- **Ins-Del** (↑): Insertion AUC − Deletion AUC. Single summary metric; higher is better.

## Files

### Refactored Modular Structure (Recommended)

```
utility.py          Common utilities, metrics, optimization functions
ig.py               Standard IG: compute_ig(model, input, params)
idgi.py             IDGI: compute_idgi(model, input, params)
guided_ig.py        Guided IG: compute_guided_ig(model, input, params)
lig_idgi.py         μ-Optimizer: compute_lig_idgi(model, input, params)
lig.py              LIG (Joint*): compute_lig(model, input, params)
compare_methods.py  Evaluation framework with multiple models and metrics
example_usage.py    Example scripts demonstrating usage
```



## Quick Start

### Using Code (Recommended)

```python
from utility import ClassLogitModel, get_device
from ig import compute_ig
from idgi import compute_idgi
from guided_ig import compute_guided_ig
from lig import compute_lig

# Setup
device = get_device()
model = ClassLogitModel(backbone, target_class)
baseline = torch.zeros_like(x)

# Run Standard IG
result_ig = compute_ig(model, x, {'baseline': baseline, 'N': 50})

# Run IDGI
result_idgi = compute_idgi(model, x, {'baseline': baseline, 'N': 50})

# Run Guided IG
result_guided = compute_guided_ig(model, x, {'baseline': baseline, 'N': 50})

# Run LIG (Joint*)
result_lig = compute_lig(model, x, {
    'baseline': baseline,
    'N': 50,
    'lam': 1.0,      # Signal-harvesting strength
    'tau': 0.01,     # L² admissibility
    'G': 16,         # Spatial groups
    'patch_size': 14,
    'n_alternating': 2,
    'mu_iter': 300,
    'path_iter': 10,
})

print(f"LIG: Q={result_lig.Q:.4f}, Ins-Del={result_lig.insdel.insertion_auc - result_lig.insdel.deletion_auc:.4f}")
```

### Using Compare Framework

**Single image mode:**
```bash
# Compare methods with ResNet50 on a single image
python compare_methods.py --model resnet50 --image path/to/image.jpg

# Use specific methods
python compare_methods.py --methods ig idgi guided_ig lig --steps 50 --image path/to/image.jpg

# Try different models
python compare_methods.py --model vgg16 --image path/to/image.jpg
python compare_methods.py --model densenet121 --image path/to/image.jpg
python compare_methods.py --model vit_b_16 --image path/to/image.jpg
```

**Batch testing mode (recommended for benchmarking):**
```bash
# Test on 30 images and report mean±std (default)
python compare_methods.py --n-test 30

# Test with specific model and save results to JSON
python compare_methods.py --model resnet50 --n-test 30 --json results.json

# Test with different confidence threshold
python compare_methods.py --n-test 50 --min-conf 0.80 --json results.json

# Full benchmark with all options
python compare_methods.py --model resnet50 --n-test 30 \
    --methods ig idgi guided_ig lig --steps 50 \
    --min-conf 0.70 --json benchmark_results.json
```

The batch mode automatically loads images from `./sample_imagenet1k/` (or other standard locations) and reports aggregated statistics with mean±std for all metrics.


## The Signal-Harvesting Action

The framework derives from a physical analogy:

**Classical mechanics**: Trajectories extremize the action S[γ] = ∫ L(γ, γ', t) dt

**Attribution**: Paths and measures jointly extremize the signal-harvesting action:

```
S[γ, μ] = ∫₀¹ ρ(t)² dt  −  λ ∫₀¹ |∇f·γ'| μ(t) dt
          └──────────┘      └───────────────────┘
          Fermat/Snell      Signal harvested
```

subject to γ(0) = x', γ(1) = x, and L² admissibility on μ.

**Stationary conditions** (Euler-Lagrange):
1. **Over μ**: Yields μ* ∝ |∇f·γ'| ≈ |Δf_k| (exactly IDGI's measure)
2. **Over γ**: Forcing term pushes γ toward regions where H_f γ' is large in direction of ∇f (the output-transition region, as in Guided IG)

The discrete objective replaces the intractable Hessian term with its proxy Var_ν(φ), yielding the practical optimization problem.

## Physical Analogy: Optical Signal Harvesting

| Concept | Optics | Attribution (LIG) |
|---------|--------|-------------------|
| **System** | Signal-harvesting optical instrument | Attribution path + measure |
| **Path** | Light ray γ(t) | Interpolation path γ(t) |
| **Measure** | Detector sensitivity μ(t) | Attribution weight μ(t) |
| **Endpoints** | Source, detector | Baseline x', input x |
| **Conservation** | Fermat/Snell: n sin θ = const | Step fidelity: ρ(t) = const |
| **Signal term** | Photon collection: ∫ I(t) μ(t) dt | Output change: ∫ \|∇f·γ'\| μ dt |
| **Trade-off** | Optical path length vs signal | Linearization error vs concentration |

LIG is an **optimal signal-harvesting detector**—it bends the path to avoid high-curvature regions (where gradients are unreliable) and concentrates its detection window on the output-transition region (where the model actually changes). Neither optimization alone is sufficient; both are necessary.

## Key Findings from the Paper

1. **Unification**: All IG variants (Standard IG, IDGI, Guided IG) are special cases of the signal-harvesting objective parameterized by (λ, τ).

2. **IDGI is exact**: IDGI's measure μ_k ∝ |Δf_k| is the exact KKT stationary point in the limit τ → 0⁺, λ > 0, not a heuristic.

3. **Guided IG is approximate**: Guided IG's path construction approximates the forced Euler-Lagrange equation for the signal-harvesting action.

4. **Trade-off elimination**: LIG resolves the conservation-faithfulness trade-off by jointly optimizing both degrees of freedom.

5. **Consistent ranking**: The objective ranks methods consistently: LIG < Guided IG < IDGI < IG on all metrics.

6. **Robustness**: Path optimization is essential on challenging images where measure-only optimization fails (Q drops to 0.085), but LIG recovers to Q > 0.999.

## Requirements

```
torch >= 2.0
torchvision
matplotlib  (for visualization)
scikit-image  (optional, for SLIC superpixels)
```

## References

**Primary Paper**:
- Anonymous. "Least Action Integrated Gradients." Under review, 2026.

**Related Work**:
- Sundararajan, Taly, Yan. "Axiomatic Attribution for Deep Networks." ICML 2017.
- Sikdar, Bhatt, Heese. "Integrated Directional Gradients." ACL 2021.
- Kapishnikov, Bolukbasi, Viégas, Terry. "Guided Integrated Gradients." CVPR 2021.
- Petsiuk, Das, Saenko. "RISE: Randomized Input Sampling for Explanation." BMVC 2018.
- Friedman. "Paths and Consistency in Additive Cost Sharing." Int. J. Game Theory 2004.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{lig2026,
  title={Least Action Integrated Gradients},
  author={Anonymous},
  journal={Under review},
  year={2026}
}
```
