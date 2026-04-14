"""
compare_methods.py - Compare Attribution Methods
=================================================

This module provides a framework to compare different attribution methods
(IG, IDGI, Guided IG, LIG) across different models and evaluation metrics.

Default methods: IG, IDGI, Guided IG, LIG
Default metrics: Insertion, Deletion, Ins-Del
Model options: ResNet50, VGG16, DenseNet121, ViT-B-16

Usage:
    python compare_methods.py --model resnet50 --image path/to/image.jpg
    python compare_methods.py --methods ig idig guided_ig lig --metrics insertion deletion
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# Import attribution methods
from ig import compute_ig
from idig import compute_idig
from guided_ig import compute_guided_ig
from lig import compute_lig
from lig_idig import compute_lig_idig

# Import utilities
from utility import (
    ClassLogitModel,
    AttributionResult,
    compute_insertion_deletion,
    get_device,
    set_seed,
    load_image,
)


# ═════════════════════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_model(model_name: str, device: torch.device) -> nn.Module:
    """
    Load a pretrained model.

    Args:
        model_name: One of 'resnet50', 'vgg16', 'densenet121', 'vit_b_16'
        device: Device to load model on

    Returns:
        Pretrained model in eval mode
    """
    model_name = model_name.lower()

    print(f"Loading {model_name}...")

    if model_name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif model_name == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif model_name == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unknown model: {model_name}. "
                         f"Choose from: resnet50, vgg16, densenet121, vit_b_16")

    model = model.to(device)
    model.eval()
    print(f"✓ {model_name} loaded")
    return model


# ═════════════════════════════════════════════════════════════════════════════
# IMAGE PREPROCESSING
# ═════════════════════════════════════════════════════════════════════════════

def load_and_preprocess_image(image_path: str, device: torch.device) -> torch.Tensor:
    """
    Load and preprocess an image for model input.

    Args:
        image_path: Path to image file
        device: Device to load image on

    Returns:
        Preprocessed image tensor (1, 3, 224, 224)
    """
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor


def create_baseline(x: torch.Tensor, baseline_type: str = 'zero') -> torch.Tensor:
    """
    Create a baseline for attribution.

    Args:
        x: Input tensor (1, C, H, W)
        baseline_type: Type of baseline ('zero', 'mean', or 'black')

    Returns:
        Baseline tensor (1, C, H, W)
    """
    if baseline_type == 'zero':
        # Zero in normalized space
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (torch.zeros_like(x) - mean) / std
    elif baseline_type == 'black':
        return torch.zeros_like(x)
    elif baseline_type == 'mean':
        # ImageNet mean in normalized space (approximately 0)
        return torch.zeros_like(x)
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")


# ═════════════════════════════════════════════════════════════════════════════
# METHOD COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def run_method(
    method_name: str,
    model: nn.Module,
    x: torch.Tensor,
    baseline: torch.Tensor,
    N: int = 50,
) -> AttributionResult:
    """
    Run a specific attribution method.

    Args:
        method_name: Name of method ('ig', 'idig', 'guided_ig', 'lig', 'lig_idig')
        model: Wrapped model (ClassLogitModel)
        x: Input tensor
        baseline: Baseline tensor
        N: Number of steps

    Returns:
        AttributionResult
    """
    params = {
        'baseline': baseline,
        'N': N,
    }

    if method_name == 'ig':
        return compute_ig(model, x, params)
    elif method_name == 'idig':
        return compute_idig(model, x, params)
    elif method_name == 'guided_ig':
        return compute_guided_ig(model, x, params)
    elif method_name == 'lig':
        params.update({
            'lam': 1.0,
            'tau': 0.01,
            'G': 16,
            'patch_size': 14,
            'n_alternating': 2,
            'mu_iter': 300,
            'path_iter': 10,
        })
        return compute_lig(model, x, params)
    elif method_name == 'lig_idig':
        params.update({
            'lam': 1.0,
            'tau': 0.01,
            'n_iter': 300,
        })
        return compute_lig_idig(model, x, params)
    else:
        raise ValueError(f"Unknown method: {method_name}")


def compare_methods_batch(
    model_name: str = 'resnet50',
    methods: list[str] = None,
    metrics: list[str] = None,
    N: int = 50,
    n_test: int = 30,
    min_conf: float = 0.70,
    device: Optional[str] = None,
    seed: int = 42,
    json_path: Optional[str] = None,
):
    """
    Compare attribution methods on multiple images with mean±std reporting.

    Args:
        model_name: Model to use ('resnet50', 'vgg16', 'densenet121', 'vit_b_16')
        methods: List of methods to compare
        metrics: List of metrics to compute
        N: Number of interpolation steps
        n_test: Number of test images
        min_conf: Minimum classification confidence
        device: Device to use (None for auto)
        seed: Random seed for reproducibility
        json_path: Path to save JSON results (optional)

    Returns:
        Dictionary with aggregated results
    """
    import numpy as np
    import json as json_module

    # Set random seed for reproducibility
    set_seed(seed)

    # Default methods and metrics
    if methods is None:
        methods = ['ig', 'idig', 'guided_ig', 'lig']
    if metrics is None:
        metrics = ['insertion', 'deletion', 'ins-del']

    # Setup device
    dev = get_device(force=device)

    # Load model once
    backbone = load_model(model_name, dev)

    print(f"\n{'='*70}")
    print(f"Batch Testing: {n_test} images, {len(methods)} methods, N={N} steps")
    print(f"{'='*70}\n")

    # Storage for all results
    all_results = {method: {
        'Q': [], 'Var_nu': [], 'CV2': [],
        'insertion_auc': [], 'deletion_auc': [], 'ins_del': [],
        'time': []
    } for method in methods}

    image_info_list = []

    # Run on n_test images
    from utility import load_image_batch
    image_batch = load_image_batch(backbone, dev, n=n_test, min_conf=min_conf)
    image_batch = image_batch[:n_test]  # Ensure we only take n_test images even if more are available
    for img_idx, (x, target_class, confidence, source, class_name) in enumerate(image_batch):

        print(f"\n--- Image {img_idx + 1}/{n_test} ---")

        image_info_list.append({
            'index': img_idx,
            'source': source,
            'target_class': int(target_class),
            'confidence': float(confidence),
            'class_name': class_name,
        })

        # Wrap model
        model = ClassLogitModel(backbone, target_class)
        baseline = torch.zeros_like(x)

        # Run each method
        for method_name in methods:
            try:
                result = run_method(method_name, model, x, baseline, N=N)

                # Store metrics
                all_results[method_name]['Q'].append(result.Q)
                all_results[method_name]['Var_nu'].append(result.Var_nu)
                all_results[method_name]['CV2'].append(result.CV2)
                all_results[method_name]['time'].append(result.elapsed_s)

                # Compute insertion/deletion if requested
                if any(m in metrics for m in ['insertion', 'deletion', 'ins-del']):
                    scores = compute_insertion_deletion(
                        model, x, baseline, result.attributions, n_steps=100, batch_size=16)
                    all_results[method_name]['insertion_auc'].append(scores.insertion_auc)
                    all_results[method_name]['deletion_auc'].append(scores.deletion_auc)
                    all_results[method_name]['ins_del'].append(
                        scores.insertion_auc - scores.deletion_auc)

                print(f"  {method_name:>12}: Q={result.Q:.4f}, time={result.elapsed_s:.2f}s")

            except Exception as e:
                print(f"  {method_name:>12}: FAILED - {e}")

    # Compute statistics
    print(f"\n{'='*70}")
    print(f"RESULTS (mean ± std over {n_test} images)")
    print(f"{'='*70}\n")

    stats = {}
    for method_name in methods:
        method_stats = {}
        for metric_name, values in all_results[method_name].items():
            if len(values) > 0:
                values_array = np.array(values)
                method_stats[metric_name] = {
                    'mean': float(np.mean(values_array)),
                    'std': float(np.std(values_array)),
                    'min': float(np.min(values_array)),
                    'max': float(np.max(values_array)),
                }
        stats[method_name] = method_stats

    # Print results table
    print(f"{'Method':<16} {'Q ↑':>15} {'Var_ν ↓':>15} {'Ins AUC ↑':>15} "
          f"{'Del AUC ↓':>15} {'Ins-Del ↑':>15} {'Time(s)':>12}")
    print("─" * 110)

    for method_name in methods:
        if method_name in stats:
            s = stats[method_name]
            print(f"{method_name:<16} "
                  f"{s['Q']['mean']:>6.4f}±{s['Q']['std']:<6.4f} "
                  f"{s['Var_nu']['mean']:>6.4f}±{s['Var_nu']['std']:<6.4f} "
                  f"{s['insertion_auc']['mean']:>6.4f}±{s['insertion_auc']['std']:<6.4f} "
                  f"{s['deletion_auc']['mean']:>6.4f}±{s['deletion_auc']['std']:<6.4f} "
                  f"{s['ins_del']['mean']:>6.4f}±{s['ins_del']['std']:<6.4f} "
                  f"{s['time']['mean']:>7.2f}±{s['time']['std']:<4.2f}")

    print(f"\n{'='*70}\n")

    # Save to JSON if requested
    if json_path:
        output = {
            'config': {
                'model': model_name,
                'n_test': n_test,
                'N': N,
                'min_conf': min_conf,
                'methods': methods,
                'metrics': metrics,
                'seed': seed,
            },
            'statistics': stats,
            'images': image_info_list,
        }
        with open(json_path, 'w') as f:
            json_module.dump(output, f, indent=2)
        print(f"✓ Results saved to {json_path}\n")

    return stats


def compare_methods(
    model_name: str = 'resnet50',
    image_path: Optional[str] = None,
    target_class: Optional[int] = None,
    methods: list[str] = None,
    metrics: list[str] = None,
    N: int = 50,
    device: Optional[str] = None,
    seed: int = 42,
):
    """
    Compare attribution methods on a given image.

    Args:
        model_name: Model to use ('resnet50', 'vgg16', 'densenet121', 'vit_b_16')
        image_path: Path to input image
        target_class: Target class index (if None, uses predicted class)
        methods: List of methods to compare
        metrics: List of metrics to compute
        N: Number of interpolation steps
        device: Device to use (None for auto)
        seed: Random seed for reproducibility (default: 42)
    """
    # Set random seed for reproducibility
    set_seed(seed)

    # Default methods and metrics
    if methods is None:
        methods = ['ig', 'idig', 'guided_ig', 'lig']
    if metrics is None:
        metrics = ['insertion', 'deletion', 'ins-del']

    # Setup device
    dev = get_device(force=device)

    # Load model
    backbone = load_model(model_name, dev)

    # Load and preprocess image
    if image_path is None:
        # Use a default image or create a random one for testing
        print("No image provided. Using random tensor for testing.")
        x = torch.randn(1, 3, 224, 224, device=dev)
    else:
        x = load_and_preprocess_image(image_path, dev)

    # Get target class
    if target_class is None:
        with torch.no_grad():
            logits = backbone(x)
            target_class = int(logits.argmax(dim=1))
        print(f"Target class (predicted): {target_class}")
    else:
        print(f"Target class (specified): {target_class}")

    # Wrap model
    model = ClassLogitModel(backbone, target_class)

    # Create baseline
    baseline = create_baseline(x, baseline_type='zero')

    # Run methods
    print(f"\n{'='*70}")
    print(f"Running {len(methods)} methods with N={N} steps...")
    print(f"{'='*70}\n")

    results = []
    for method_name in methods:
        print(f"Running {method_name.upper()}...")
        try:
            result = run_method(method_name, model, x, baseline, N=N)
            results.append(result)
            print(f"✓ {method_name.upper()}: Q={result.Q:.4f}, "
                  f"CV²={result.CV2:.4f}, time={result.elapsed_s:.2f}s\n")
        except Exception as e:
            print(f"✗ {method_name.upper()} failed: {e}\n")

    # Compute insertion/deletion metrics if requested
    if any(m in metrics for m in ['insertion', 'deletion', 'ins-del']):
        print(f"\n{'='*70}")
        print("Computing Insertion/Deletion metrics...")
        print(f"{'='*70}\n")

        print(f"{'Method':<16} {'Ins AUC':>10} {'Del AUC':>10} {'Ins-Del':>10} "
              f"{'Q':>8} {'Time(s)':>8}")
        print("─" * 70)

        for result in results:
            scores = compute_insertion_deletion(
                model, x, baseline, result.attributions, n_steps=100, batch_size=16)
            result.insdel = scores
            ins_del_diff = scores.insertion_auc - scores.deletion_auc
            print(f"{result.name:<16} {scores.insertion_auc:>10.4f} "
                  f"{scores.deletion_auc:>10.4f} {ins_del_diff:>10.4f} "
                  f"{result.Q:>8.4f} {result.elapsed_s:>8.2f}")

    print(f"\n{'='*70}")
    print("Comparison complete!")
    print(f"{'='*70}\n")

    return results


# ═════════════════════════════════════════════════════════════════════════════
# COMMAND LINE INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Compare attribution methods across models and metrics'
    )

    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'vgg16', 'densenet121', 'vit_b_16'],
                        help='Model to use (default: resnet50)')

    parser.add_argument('--image', type=str, default=None,
                        help='Path to input image (optional)')

    parser.add_argument('--target-class', type=int, default=None,
                        help='Target class index (default: use predicted class)')

    parser.add_argument('--methods', type=str, nargs='+',
                        default=['ig', 'idig', 'guided_ig', 'lig'],
                        choices=['ig', 'idig', 'guided_ig', 'lig', 'lig_idig'],
                        help='Methods to compare (default: ig idig guided_ig lig)')

    parser.add_argument('--metrics', type=str, nargs='+',
                        default=['insertion', 'deletion', 'ins-del'],
                        choices=['insertion', 'deletion', 'ins-del'],
                        help='Metrics to compute (default: insertion deletion ins-del)')

    parser.add_argument('--steps', type=int, default=50,
                        help='Number of interpolation steps N (default: 50)')

    parser.add_argument('--device', type=str, default=None,
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to use (default: auto)')

    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    parser.add_argument('--n-test', type=int, default=None,
                        help='Number of test images for batch testing (default: None for single image)')

    parser.add_argument('--min-conf', type=float, default=0.70,
                        help='Minimum classification confidence for batch testing (default: 0.70)')

    parser.add_argument('--json', type=str, default=None,
                        help='Path to save JSON results (for batch testing)')

    args = parser.parse_args()

    # Batch testing mode
    if args.n_test is not None:
        results = compare_methods_batch(
            model_name=args.model,
            methods=args.methods,
            metrics=args.metrics,
            N=args.steps,
            n_test=args.n_test,
            min_conf=args.min_conf,
            device=args.device,
            seed=args.seed,
            json_path=args.json,
        )
        return

    # Single image mode
    # Validate image path
    if args.image is not None and not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return

    # Run comparison
    results = compare_methods(
        model_name=args.model,
        image_path=args.image,
        target_class=args.target_class,
        methods=args.methods,
        metrics=args.metrics,
        N=args.steps,
        device=args.device,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
