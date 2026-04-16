"""
example_usage.py - Example usage of refactored attribution methods
===================================================================

This script demonstrates how to use the refactored attribution methods.
"""

import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image

# Import methods
from ig import compute_ig
from idgi import compute_idgi
from guided_ig import compute_guided_ig
from lig import compute_lig
from lig_idgi import compute_lig_idgi

# Import utilities
from utility import ClassLogitModel, get_device, compute_insertion_deletion, set_seed


def example_basic():
    """Basic example with a random image."""
    print("="*70)
    print("EXAMPLE 1: Basic usage with random image")
    print("="*70 + "\n")

    # Set seed for reproducibility
    set_seed(42)

    # Setup device
    device = get_device()

    # Load a pretrained model
    print("Loading ResNet50...")
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = backbone.to(device)
    backbone.eval()

    # Create random image for testing
    x = torch.randn(1, 3, 224, 224, device=device)
    baseline = torch.zeros_like(x)

    # Get predicted class
    with torch.no_grad():
        logits = backbone(x)
        target_class = int(logits.argmax(dim=1))
    print(f"Target class: {target_class}\n")

    # Wrap model
    model = ClassLogitModel(backbone, target_class)

    # Define parameters
    params = {'baseline': baseline, 'N': 50}

    # Run methods
    print("Running attribution methods...\n")

    print("1. Standard IG...")
    result_ig = compute_ig(model, x, params)
    print(f"   Q={result_ig.Q:.4f}, CV²={result_ig.CV2:.4f}, "
          f"time={result_ig.elapsed_s:.2f}s\n")

    print("2. IDGI...")
    result_idgi = compute_idgi(model, x, params)
    print(f"   Q={result_idgi.Q:.4f}, CV²={result_idgi.CV2:.4f}, "
          f"time={result_idgi.elapsed_s:.2f}s\n")

    print("3. Guided IG...")
    result_guided = compute_guided_ig(model, x, params)
    print(f"   Q={result_guided.Q:.4f}, CV²={result_guided.CV2:.4f}, "
          f"time={result_guided.elapsed_s:.2f}s\n")

    print("4. LIG-IDIG (μ-Optimizer)...")
    params_lig_idgi = {
        'baseline': baseline,
        'N': 50,
        'lam': 1.0,
        'tau': 0.01,
        'n_iter': 300,
    }
    result_lig_idgi = compute_lig_idgi(model, x, params_lig_idgi)
    print(f"   Q={result_lig_idgi.Q:.4f}, CV²={result_lig_idgi.CV2:.4f}, "
          f"time={result_lig_idgi.elapsed_s:.2f}s\n")

    print("Done!\n")


def example_with_evaluation():
    """Example with insertion/deletion evaluation."""
    print("="*70)
    print("EXAMPLE 2: With insertion/deletion evaluation")
    print("="*70 + "\n")

    # Set seed for reproducibility
    set_seed(42)

    # Setup device
    device = get_device()

    # Load model
    print("Loading ResNet50...")
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = backbone.to(device)
    backbone.eval()

    # Create random image
    x = torch.randn(1, 3, 224, 224, device=device)
    baseline = torch.zeros_like(x)

    # Get predicted class
    with torch.no_grad():
        logits = backbone(x)
        target_class = int(logits.argmax(dim=1))
    print(f"Target class: {target_class}\n")

    # Wrap model
    model = ClassLogitModel(backbone, target_class)

    # Run methods
    methods = ['ig', 'idgi', 'guided_ig']
    results = []

    print("Running methods...\n")
    for method_name in methods:
        params = {'baseline': baseline, 'N': 50}

        if method_name == 'ig':
            result = compute_ig(model, x, params)
        elif method_name == 'idgi':
            result = compute_idgi(model, x, params)
        elif method_name == 'guided_ig':
            result = compute_guided_ig(model, x, params)

        results.append(result)
        print(f"{method_name.upper()}: Q={result.Q:.4f}")

    # Compute insertion/deletion
    print("\n" + "="*70)
    print("Computing Insertion/Deletion metrics...")
    print("="*70 + "\n")

    print(f"{'Method':<16} {'Ins AUC':>10} {'Del AUC':>10} {'Ins-Del':>10} {'Q':>8}")
    print("─" * 60)

    for result in results:
        scores = compute_insertion_deletion(
            model, x, baseline, result.attributions, n_steps=50, batch_size=16)
        result.insdel = scores
        ins_del_diff = scores.insertion_auc - scores.deletion_auc
        print(f"{result.name:<16} {scores.insertion_auc:>10.4f} "
              f"{scores.deletion_auc:>10.4f} {ins_del_diff:>10.4f} {result.Q:>8.4f}")

    print("\nDone!\n")


def example_lig():
    """Example with LIG (Joint* method)."""
    print("="*70)
    print("EXAMPLE 3: LIG (Joint* method)")
    print("="*70 + "\n")

    # Set seed for reproducibility
    set_seed(42)

    # Setup device
    device = get_device()

    # Load model
    print("Loading ResNet50...")
    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    backbone = backbone.to(device)
    backbone.eval()

    # Create random image
    x = torch.randn(1, 3, 224, 224, device=device)
    baseline = torch.zeros_like(x)

    # Get predicted class
    with torch.no_grad():
        logits = backbone(x)
        target_class = int(logits.argmax(dim=1))
    print(f"Target class: {target_class}\n")

    # Wrap model
    model = ClassLogitModel(backbone, target_class)

    # Run LIG
    print("Running LIG (Joint* method)...")
    params_lig = {
        'baseline': baseline,
        'N': 50,
        'lam': 1.0,
        'tau': 0.01,
        'G': 16,
        'patch_size': 14,
        'n_alternating': 2,
        'mu_iter': 300,
        'path_iter': 10,
    }

    result_lig = compute_lig(model, x, params_lig)

    print(f"\nResults:")
    print(f"  Q: {result_lig.Q:.4f}")
    print(f"  CV²: {result_lig.CV2:.4f}")
    print(f"  Var_nu: {result_lig.Var_nu:.4f}")
    print(f"  Time: {result_lig.elapsed_s:.2f}s")

    # Show optimization history
    if result_lig.Q_history:
        print(f"\n  Optimization history:")
        for entry in result_lig.Q_history:
            print(f"    Iter {entry['iteration']}: "
                  f"Q_mu={entry['Q_after_mu']:.4f}, "
                  f"Q_path={entry['Q_after_path']:.4f}, "
                  f"best_Q={entry['best_Q']:.4f}")

    print("\nDone!\n")


if __name__ == '__main__':
    # Run examples
    example_basic()
    print("\n" + "="*70 + "\n")
    example_with_evaluation()
    print("\n" + "="*70 + "\n")
    example_lig()
