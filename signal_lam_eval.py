"""
signal_lam_eval.py — Multi-Image Evaluation for Signal-Harvesting IG
======================================================================

Evaluates all IG methods on multiple images and reports statistics.

Usage:
    python signal_lam_eval.py --n 100 --image-folder ./sample_imagenet1k
    python signal_lam_eval.py --n 50 --steps 50 --lam 1.0
"""

from __future__ import annotations

import os
import time
import argparse
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import numpy as np
from PIL import Image

# Import from signal_lam.py
from signal_lam import (
    run_all_methods,
    compute_signal_harvesting_objective,
    _forward_scalar,
)
from lam import ClassLogitModel
from utilss import get_device, set_seed


def load_images_from_folder(
    folder_path: str,
    n_images: int,
    min_conf: float,
    device: torch.device,
    backbone: nn.Module,
) -> list[tuple[torch.Tensor, torch.Tensor, dict]]:
    """
    Load n images from folder that meet confidence threshold.

    Returns:
        List of (x, baseline, info) tuples
    """
    # Image preprocessing transform
    tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Find all images in folder
    folder = Path(folder_path)
    if not folder.exists():
        raise ValueError(f"Folder not found: {folder_path}")

    image_files = sorted([
        f for f in folder.iterdir()
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png']
    ])

    if len(image_files) == 0:
        raise ValueError(f"No images found in {folder_path}")

    print(f"Found {len(image_files)} images in {folder_path}")

    # Load ImageNet class names
    try:
        import urllib.request
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        with urllib.request.urlopen(url, timeout=5) as f:
            class_names = [line.decode().strip() for line in f]
    except Exception:
        class_names = [f"class_{i}" for i in range(1000)]

    # Baseline (black image in normalized space)
    baseline_raw = torch.zeros(1, 3, 224, 224, device=device)

    loaded_images = []
    skipped = 0

    for img_file in image_files:
        if len(loaded_images) >= n_images:
            break

        try:
            # Load and preprocess image
            img = Image.open(img_file).convert("RGB")
            x = tf(img).unsqueeze(0).to(device)

            # Get prediction
            with torch.no_grad():
                logits = backbone(x)
                probs = torch.softmax(logits, dim=1)
                conf, pred_class = probs.max(dim=1)

            conf = conf.item()
            pred_class = pred_class.item()

            # Check confidence threshold
            if conf < min_conf:
                skipped += 1
                continue

            # Wrap model for this predicted class
            model = ClassLogitModel(backbone, pred_class)

            # Create info dict
            info = {
                "model": "ResNet-50",
                "source": str(img_file.name),
                "target_class": class_names[pred_class] if pred_class < len(class_names) else f"class_{pred_class}",
                "confidence": conf,
                "pred_class_idx": pred_class,
            }

            loaded_images.append((x, baseline_raw, info, model))
            print(f"  [{len(loaded_images)}/{n_images}] {img_file.name}: {info['target_class']} (conf={conf:.4f})")

        except Exception as e:
            print(f"  Skipped {img_file.name}: {e}")
            skipped += 1
            continue

    if len(loaded_images) == 0:
        raise ValueError(f"No images with confidence >= {min_conf} found")

    print(f"\nLoaded {len(loaded_images)} images (skipped {skipped})")
    return loaded_images


def run_evaluation_on_images(
    images: list,
    N: int,
    lam: float,
    tau: float,
    guided_init: bool,
    device: torch.device,
) -> dict:
    """
    Run all methods on multiple images and compute statistics.

    Returns:
        dict with mean/std for each metric per method
    """
    n_images = len(images)

    # Initialize accumulators for each method
    # Methods: IG, IDGI, Guided IG, μ-Optimized*, Joint(λ=0), Joint*
    method_names = ["IG", "IDGI", "Guided IG", "μ-Optimized*", "Joint(λ=0)", "Joint*"]
    stats = {name: {
        "Q": [],
        "Var_nu": [],
        "CV2": [],
        "Obj": [],
        "Time": [],
    } for name in method_names}

    # Run on each image
    for idx, (x, baseline, info, model) in enumerate(images):
        print(f"\n{'='*70}")
        print(f"Image {idx+1}/{n_images}: {info['source']}")
        print(f"Class: {info['target_class']} (conf={info['confidence']:.4f})")
        print(f"{'='*70}")

        # Compute delta_f
        f_x = _forward_scalar(model, x)
        f_bl = _forward_scalar(model, baseline)
        delta_f = f_x - f_bl

        print(f"f(x) = {f_x:.4f}, f(bl) = {f_bl:.4f}, Δf = {delta_f:.4f}")

        # Run all methods
        try:
            methods = run_all_methods(
                model, x, baseline, N=N,
                lam=lam, tau=tau,
                guided_init=guided_init
            )

            # Extract metrics for each method
            for m in methods:
                d_arr = torch.tensor([s.d_k for s in m.steps], device=device)
                df_arr = torch.tensor([s.delta_f_k for s in m.steps], device=device)
                mu_arr = torch.tensor([s.mu_k for s in m.steps], device=device)
                obj, _, _, _ = compute_signal_harvesting_objective(
                    d_arr, df_arr, mu_arr, lam=lam, tau=tau
                )

                # Store metrics
                stats[m.name]["Q"].append(m.Q)
                stats[m.name]["Var_nu"].append(m.Var_nu)
                stats[m.name]["CV2"].append(m.CV2)
                stats[m.name]["Obj"].append(obj)
                stats[m.name]["Time"].append(m.elapsed_s)

                print(f"{m.name:<16} Q={m.Q:.6f} Var_ν={m.Var_nu:.6e} Time={m.elapsed_s:.1f}s")

        except Exception as e:
            print(f"ERROR on image {idx+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Compute statistics
    results = {}
    for name in method_names:
        if len(stats[name]["Q"]) == 0:
            continue

        results[name] = {
            "Q_mean": float(np.mean(stats[name]["Q"])),
            "Q_std": float(np.std(stats[name]["Q"])),
            "Var_nu_mean": float(np.mean(stats[name]["Var_nu"])),
            "Var_nu_std": float(np.std(stats[name]["Var_nu"])),
            "CV2_mean": float(np.mean(stats[name]["CV2"])),
            "CV2_std": float(np.std(stats[name]["CV2"])),
            "Obj_mean": float(np.mean(stats[name]["Obj"])),
            "Obj_std": float(np.std(stats[name]["Obj"])),
            "Time_mean": float(np.mean(stats[name]["Time"])),
            "Time_std": float(np.std(stats[name]["Time"])),
            "n_images": len(stats[name]["Q"]),
        }

    return results


def print_results_table(results: dict):
    """Print formatted results table with mean ± std."""
    print("\n" + "="*100)
    print("EVALUATION RESULTS (Mean ± Std)")
    print("="*100)

    header = f"{'Method':<16} {'Q':>18} {'Var_ν':>18} {'CV²':>18} {'Obj':>18} {'Time':>12} {'N':>5}"
    print(header)
    print("-"*100)

    for method_name, metrics in results.items():
        q_str = f"{metrics['Q_mean']:.4f}±{metrics['Q_std']:.4f}"
        var_str = f"{metrics['Var_nu_mean']:.2e}±{metrics['Var_nu_std']:.2e}"
        cv2_str = f"{metrics['CV2_mean']:.4f}±{metrics['CV2_std']:.4f}"
        obj_str = f"{metrics['Obj_mean']:.3f}±{metrics['Obj_std']:.3f}"
        time_str = f"{metrics['Time_mean']:.1f}±{metrics['Time_std']:.1f}s"
        n_str = f"{metrics['n_images']}"

        print(f"{method_name:<16} {q_str:>18} {var_str:>18} {cv2_str:>18} "
              f"{obj_str:>18} {time_str:>12} {n_str:>5}")

    print("="*100)


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Image Evaluation for Signal-Harvesting IG")

    # Evaluation parameters
    parser.add_argument("--n", type=int, default=1,
                        help="Number of images to evaluate (default: 1)")
    parser.add_argument("--image-folder", type=str, default=None,
                        help="Folder containing images (default: auto-detect sample_imagenet1k)")

    # Method parameters (same as signal_lam.py)
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of interpolation steps N")
    parser.add_argument("--lam", type=float, default=1.0,
                        help="Signal-harvesting strength λ")
    parser.add_argument("--tau", type=float, default=0.01,
                        help="L2 admissibility multiplier τ")
    parser.add_argument("--min-conf", type=float, default=0.70,
                        help="Minimum confidence threshold for images")
    parser.add_argument("--guided-init", action="store_true",
                        help="Initialize Joint methods from Guided IG")

    # Output parameters
    parser.add_argument("--json", type=str, default=None,
                        help="Export results to JSON file")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    # Set random seed
    set_seed(args.seed)

    # Get device
    device = get_device(force=args.device)
    print(f"Using device: {device}")

    # Determine image folder
    if args.image_folder is None:
        # Auto-detect sample_imagenet1k
        for candidate in ["./sample_imagenet1k", "../sample_imagenet1k",
                          os.path.expanduser("~/sample_imagenet1k")]:
            if os.path.isdir(candidate):
                args.image_folder = candidate
                break

        if args.image_folder is None:
            raise ValueError("No image folder found. Use --image-folder to specify path.")

    print(f"Image folder: {args.image_folder}")
    print(f"Evaluating on {args.n} images")
    print(f"Parameters: N={args.steps}, λ={args.lam}, τ={args.tau}")

    # Load backbone model once
    print("\nLoading ResNet-50...")
    backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    backbone = backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad_(False)

    # Load images
    print(f"\nLoading images from {args.image_folder}...")
    images = load_images_from_folder(
        args.image_folder,
        n_images=args.n,
        min_conf=args.min_conf,
        device=device,
        backbone=backbone,
    )

    # Run evaluation
    print(f"\nRunning evaluation on {len(images)} images...")
    t0 = time.time()
    results = run_evaluation_on_images(
        images,
        N=args.steps,
        lam=args.lam,
        tau=args.tau,
        guided_init=args.guided_init,
        device=device,
    )
    total_time = time.time() - t0

    # Print results table
    print_results_table(results)
    print(f"\nTotal evaluation time: {total_time:.1f}s ({total_time/len(images):.1f}s per image)")

    # Export to JSON if requested
    if args.json:
        output = {
            "config": {
                "n_images": len(images),
                "N": args.steps,
                "lam": args.lam,
                "tau": args.tau,
                "min_conf": args.min_conf,
                "guided_init": args.guided_init,
                "image_folder": args.image_folder,
                "device": str(device),
                "seed": args.seed,
            },
            "results": results,
            "total_time": total_time,
        }

        with open(args.json, "w") as f:
            json.dump(output, f, indent=2)

        print(f"\nResults exported to {args.json}")


if __name__ == "__main__":
    main()
