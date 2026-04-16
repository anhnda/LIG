# Benchmark Results — Attribution Methods on ImageNet-1K

**Date:** 2026-04-15/16
**Images:** 50 per backbone (min confidence 75%, seed 42)
**Steps:** N = 50
**Methods:** IG, IDGI, Guided IG (custom), Guided IG (standard, PAIR 2021), IDGI (standard), LIG
**Hardware:** NVIDIA A100-SXM4-40GB

**Command:**
```bash
python run_benchmark.py \
  --models vit_b_16 inception_v3 swin_b convnext_base efficientnet_b0 mobilenet_v2 \
  --methods ig idgi guided_ig guided_ig_standard idgi_standard lig \
  --n-test 50 --steps 50 --min-conf 0.75 --seed 42 \
  --outdir results/full_benchmark_20260415
```

---

## Summary

| Metric | Best Method | Wins (out of 9 models) | Avg Rank |
|---|---|---|---|
| **Q (completeness)** | **LIG** | 8/9 | 1.11 |
| **Ins-Del AUC** | **LIG** | 8/9 | 1.22 |

LIG achieves the best Q in 8 out of 9 backbones (only swin_b where Guided IG wins) and the best Insertion-Deletion AUC in 8 out of 9 (only vgg16 where IDGI standard wins).

### Average Rank Across 9 Backbones

| Method | Q Rank | Ins-Del Rank |
|---|---|---|
| LIG | **1.11** | **1.22** |
| IDGI | 2.67 | 3.11 |
| Guided IG (custom) | 3.33 | 5.44 |
| Guided IG (standard) | 5.44 | 3.00 |
| IDGI (standard) | 4.33 | 3.44 |
| IG | 4.11 | 4.78 |

---

## Q — Completeness (mean +/- std)

Higher is better. Q = 1.0 means the attribution perfectly recovers the logit difference.

| Model | IG | IDGI | Guided IG | Guided IG (std) | IDGI (std) | LIG |
|---|---|---|---|---|---|---|
| resnet50 | 0.7260+/-0.2855 | 0.8698+/-0.2033 | 0.9384+/-0.1351 | 0.3403+/-0.2097 | 0.1470+/-0.2535 | **0.9657+/-0.1621** |
| vgg16 | 0.8847+/-0.1224 | 0.9235+/-0.1024 | 0.3657+/-0.3791 | 0.5839+/-0.2551 | 0.8809+/-0.1364 | **0.9914+/-0.0578** |
| densenet121 | 0.7655+/-0.1887 | 0.8244+/-0.1893 | 0.9167+/-0.1653 | 0.5990+/-0.2356 | 0.7698+/-0.2880 | **0.9793+/-0.1002** |
| vit_b_16 | 0.6716+/-0.2778 | 0.7516+/-0.2566 | 0.8956+/-0.1974 | 0.5065+/-0.2372 | 0.7131+/-0.2837 | **0.9144+/-0.2375** |
| inception_v3 | 0.7061+/-0.2102 | 0.8056+/-0.1954 | 0.0620+/-0.0844 | 0.5922+/-0.2283 | 0.8158+/-0.1373 | **0.9717+/-0.1445** |
| swin_b | 0.5748+/-0.2897 | 0.7853+/-0.2625 | **0.9672+/-0.1415** | 0.1585+/-0.2279 | 0.0829+/-0.1277 | 0.9322+/-0.2330 |
| convnext_base | 0.6902+/-0.2738 | 0.7333+/-0.2829 | 0.8151+/-0.3095 | 0.3780+/-0.2340 | 0.6177+/-0.3159 | **0.9850+/-0.0578** |
| efficientnet_b0 | 0.7445+/-0.1798 | 0.8269+/-0.1808 | 0.2664+/-0.3098 | 0.5045+/-0.2399 | 0.7391+/-0.2436 | **0.9612+/-0.1688** |
| mobilenet_v2 | 0.6240+/-0.2651 | 0.7704+/-0.2415 | 0.7365+/-0.3006 | 0.3533+/-0.1863 | 0.6777+/-0.3074 | **0.8415+/-0.3029** |

---

## Insertion-Deletion AUC (mean +/- std)

Higher is better. Ins-Del = Insertion AUC - Deletion AUC.

| Model | IG | IDGI | Guided IG | Guided IG (std) | IDGI (std) | LIG |
|---|---|---|---|---|---|---|
| resnet50 | 0.1960+/-0.2039 | 0.2926+/-0.1881 | 0.2410+/-0.1616 | 0.3002+/-0.1664 | 0.1561+/-0.1319 | **0.3341+/-0.1597** |
| vgg16 | 0.2934+/-0.2395 | 0.3631+/-0.1962 | 0.0842+/-0.1299 | 0.3177+/-0.2321 | **0.3676+/-0.1911** | 0.3408+/-0.1987 |
| densenet121 | 0.2986+/-0.2255 | 0.3318+/-0.1886 | 0.2643+/-0.1666 | 0.3498+/-0.2037 | 0.2990+/-0.1917 | **0.3591+/-0.1902** |
| vit_b_16 | 0.3136+/-0.2321 | 0.4028+/-0.1990 | 0.1416+/-0.1764 | 0.3069+/-0.2362 | 0.3926+/-0.1937 | **0.4631+/-0.2086** |
| inception_v3 | 0.4053+/-0.2401 | 0.4354+/-0.1987 | 0.3181+/-0.1910 | 0.4210+/-0.2381 | 0.4381+/-0.2038 | **0.4510+/-0.2492** |
| swin_b | 0.0268+/-0.2543 | -0.0635+/-0.2142 | 0.1019+/-0.1743 | 0.1794+/-0.2226 | -0.1030+/-0.1973 | **0.2058+/-0.2141** |
| convnext_base | 0.1326+/-0.1920 | 0.1766+/-0.1853 | 0.1170+/-0.1472 | 0.2810+/-0.1653 | 0.1849+/-0.1738 | **0.2870+/-0.1698** |
| efficientnet_b0 | 0.3017+/-0.2164 | 0.3184+/-0.1874 | 0.0737+/-0.1714 | 0.3310+/-0.2058 | 0.3362+/-0.1977 | **0.3758+/-0.1741** |
| mobilenet_v2 | 0.2441+/-0.1992 | 0.3331+/-0.1754 | 0.0810+/-0.1721 | 0.3267+/-0.1850 | 0.3088+/-0.1717 | **0.3603+/-0.1608** |

---

## Insertion AUC (mean)

Higher is better.

| Model | IG | IDGI | Guided IG | Guided IG (std) | IDGI (std) | LIG |
|---|---|---|---|---|---|---|
| resnet50 | 0.5145 | 0.5680 | 0.5248 | 0.5648 | 0.4511 | **0.6038** |
| vgg16 | 0.4628 | 0.5218 | 0.3215 | 0.4722 | **0.5205** | 0.4938 |
| densenet121 | 0.5690 | 0.5881 | 0.5390 | 0.5982 | 0.5542 | **0.6107** |
| vit_b_16 | 0.6972 | 0.7133 | 0.6157 | 0.6863 | 0.7076 | **0.7785** |
| inception_v3 | 0.6115 | 0.6209 | 0.5549 | 0.6182 | 0.6235 | **0.6335** |
| swin_b | 0.6809 | 0.6603 | 0.7033 | 0.7357 | 0.6079 | **0.7911** |
| convnext_base | 0.7119 | 0.7278 | 0.7149 | 0.7928 | 0.7373 | **0.8067** |
| efficientnet_b0 | 0.6748 | 0.6908 | 0.5356 | 0.6790 | 0.6981 | **0.7413** |
| mobilenet_v2 | 0.5298 | 0.5965 | 0.4060 | 0.5705 | 0.5731 | **0.6232** |

---

## Deletion AUC (mean)

Lower is better.

| Model | IG | IDGI | Guided IG | Guided IG (std) | IDGI (std) | LIG |
|---|---|---|---|---|---|---|
| resnet50 | 0.3184 | 0.2754 | 0.2838 | **0.2646** | 0.2950 | 0.2698 |
| vgg16 | 0.1694 | 0.1587 | 0.2372 | 0.1545 | **0.1529** | **0.1529** |
| densenet121 | 0.2703 | 0.2563 | 0.2746 | **0.2483** | 0.2552 | 0.2516 |
| vit_b_16 | 0.3836 | **0.3105** | 0.4741 | 0.3794 | 0.3149 | 0.3154 |
| inception_v3 | 0.2062 | 0.1855 | 0.2368 | 0.1972 | 0.1853 | **0.1824** |
| swin_b | 0.6541 | 0.7238 | 0.6013 | **0.5563** | 0.7109 | 0.5853 |
| convnext_base | 0.5793 | 0.5512 | 0.5979 | **0.5118** | 0.5523 | 0.5197 |
| efficientnet_b0 | 0.3731 | 0.3724 | 0.4619 | **0.3479** | 0.3619 | 0.3655 |
| mobilenet_v2 | 0.2856 | 0.2634 | 0.3250 | **0.2438** | 0.2644 | 0.2628 |

---

## Var(nu) — Measure Variance (mean)

Lower is better. Var(nu) = 0 means uniform measure (ideal for IG axioms).

| Model | IG | IDGI | Guided IG | Guided IG (std) | IDGI (std) | LIG |
|---|---|---|---|---|---|---|
| resnet50 | 0.2177 | 0.1358 | 0.4980 | 1.4315 | 0.2540 | **0.0050** |
| vgg16 | 0.1262 | 0.0668 | **0.0074** | 0.8776 | 0.0992 | 0.0053 |
| densenet121 | 0.3719 | 0.2172 | 0.7742 | 0.5206 | 0.1969 | **0.0705** |
| vit_b_16 | 1.1237 | 0.5381 | **0.0004** | 0.6311 | 0.5787 | 0.0865 |
| inception_v3 | 0.3478 | 0.1963 | **0.0045** | 0.5910 | 0.2140 | 0.0734 |
| swin_b | 0.0916 | 0.0266 | **0.0003** | 1.4874 | 0.0936 | 0.0079 |
| convnext_base | 0.5659 | 0.4122 | **0.0001** | 0.6235 | 0.4828 | 0.0040 |
| efficientnet_b0 | 0.3156 | 0.1732 | **0.0009** | 0.7116 | 0.2308 | 0.0774 |
| mobilenet_v2 | 0.9336 | 0.4555 | **0.0041** | 1.2242 | 0.5412 | 0.4308 |

---

## Time per Image (mean seconds)

| Model | IG | IDGI | Guided IG | Guided IG (std) | IDGI (std) | LIG |
|---|---|---|---|---|---|---|
| resnet50 | 0.24 | 0.24 | 1.26 | 1.41 | 1.24 | 53.56 |
| vgg16 | 0.40 | 0.39 | 0.90 | 1.02 | 0.89 | 89.83 |
| densenet121 | 0.30 | 0.30 | 3.40 | 3.55 | 3.36 | 61.14 |
| vit_b_16 | 0.42 | 0.36 | 1.44 | 1.60 | 1.41 | 78.13 |
| inception_v3 | 0.12 | 0.09 | 2.43 | 2.57 | 2.40 | 12.85 |
| swin_b | 0.53 | 0.46 | 5.27 | 5.44 | 5.21 | 91.28 |
| convnext_base | 0.47 | 0.46 | 2.47 | 2.64 | 2.44 | 93.81 |
| efficientnet_b0 | 0.10 | 0.08 | 1.93 | 2.10 | 1.90 | 11.94 |
| mobilenet_v2 | 0.06 | 0.06 | 1.25 | 1.41 | 1.23 | 9.01 |

LIG is 50-200x slower than IG/IDGI due to per-step optimization. Guided IG variants and IDGI standard are 5-15x slower than IG.

---

## Key Observations

1. **LIG dominates Q and Ins-Del AUC.** It wins 8/9 backbones on Q (avg rank 1.11) and 8/9 on Ins-Del (avg rank 1.22). The only exceptions are swin_b (Q, Guided IG wins) and vgg16 (Ins-Del, IDGI standard wins).

2. **Guided IG (custom) achieves the lowest Var(nu)** on most models, meaning its integration measure is closest to uniform. However, this does not always translate to better Q or Ins-Del.

3. **Guided IG (standard, PAIR 2021) has the lowest Deletion AUC** on most models, meaning it is best at identifying features whose removal most reduces the prediction. However, its Q scores are inconsistent.

4. **IDGI is a consistent runner-up** with avg rank 2.67 for Q and 3.11 for Ins-Del, while being as fast as IG.

5. **LIG's cost is 50-200x IG.** On large models like swin_b / convnext_base it takes ~90s per image (N=50). On lightweight models (mobilenet_v2, efficientnet_b0) it takes 9-12s.

6. **Guided IG (custom) is unstable on some architectures.** It achieves Q=0.06 on inception_v3 and Q=0.27 on efficientnet_b0, suggesting the custom path optimization can fail for certain model structures.

---

## Configuration

```
n_test:    50 images per backbone
N:         50 integration steps
min_conf:  0.75 (filter low-confidence samples)
seed:      42
dataset:   sample_imagenet1k (50 images)
metrics:   insertion, deletion, ins-del AUC
```

**Backbones:** resnet50, vgg16, densenet121 (L4 GPU), vit_b_16, inception_v3, swin_b, convnext_base, efficientnet_b0, mobilenet_v2 (A100 GPU)
