# Title Research Results — Locked Top 4 (Colab + <20GB)

Source: 5 librarian sweeps (tabular, small-vision, tiny-NLP, time-series, trustworthy). Date: 2026-10-05.
Constraints: public dataset <20GB (hard cap 40GB), Colab free-T4 minutes-hours, novel (<5 papers).

---

## #1 LOCKED WINNER — MedMNIST-C Calibration
**Title:** How Miscalibrated Are Lightweight CNNs Under Clinical Corruptions? Calibration-Aware Losses on BloodMNIST-C and DermaMNIST-C

**Gap (0 calibration papers):**
1. Yang et al. MedMNIST v2, Nature Sci Data 2023 — 12x2D+6x3D, 708k images, no corruption, no calibration-under-shift.
2. Di Salvo et al. MedMNIST-C, arXiv:2406.17536 (MICCAI-ADSMI 2024) — 12 sets x 9 modalities, 5 severities, BE/rBE only, ResNet-18 pretrained; targeted aug +9.2 AUC, leaves AugMix/multi-chain future; no ECE/NLL, no focal-family, no from-scratch small-model study.
3. Maron 2021 skin-robustness + Islam 2023 stress-testing — accuracy-only, single-modality.

**Dataset + size:** BloodMNIST 17,092 imgs 8-cls 28x28 + DermaMNIST ~10k 7-cls (or BreastMNIST 546 train extreme low-data). Via `MedMNIST/MedMNIST` pip + `francescodisalvo05/medmnistc-api` at 28/64px. Total <500MB.

**Baselines:** ResNet-18 + MobileNetV3-Small or ViT-Tiny-28px (timm); pretrained vs from-scratch (1 cell); losses CE / Focal / DualFocal; augs none / RandAugment / MedMNIST-C targeted. Metrics: AUC + balanced-acc + ECE + NLL, clean + per-category (digital/noise/blur/color/task-specific).

**Colab:** 28px train 10-20min T4 per run; 64px <30min. Full matrix 2 sets x 3 losses x 3 augs ~4-6h.

**Hypothesis:** DualFocal cuts corrupted ECE >=4pp vs CE while targeted aug gives >=+3pp corrupted AUC over RandAugment; combination stacks. Fail if ECE improves ID-only.

---

## #2 — EuroSAT-C + Cross-Dataset Shift
**Title:** Small CNNs Under Realistic Sensor Shift: Corruption Calibration and Cross-Dataset Generalization from EuroSAT

**Gap:**
1. Helber et al. EuroSAT, IEEE JSTARS 2019 — 27k Sentinel-2 patches, 10 cls, 13 bands, 98.57% clean SOTA; saturated ID, no corruption protocol.
2. Hendrycks & Dietterich 2019 + `hendrycks/robustness` — CIFAR/ImageNet-C standardized, but no EuroSAT-C (verified no Zenodo/official).
3. EarthShift 2026 / PANGAEA / CVPR25 Al-Emadi — foundation-model geographic-shift, no small-CNN synthetic-corruption + calibration. EuroSAT->RESISC45 zero-shot reports large-model accuracy-only.

**Dataset + size:** EuroSAT RGB 27k 64x64 (Zenodo 7711810 / TFDS `eurosat`, ~100-300MB). Eval A: synthetic EuroSAT-C (15 Hendrycks corruptions on-fly). Eval B: EuroSAT->NWPU-RESISC45 10-class overlap subset (~few 100MB). Total <5GB. RGB-only (MS 13-band optional stretch).

**Baselines:** ResNet-18 + MobileNetV3-Large / EfficientNet-B0; losses CE vs Focal/DualFocal; augs none vs AugMix vs RandAugment; AdamW vs SAM cell. Metrics: clean acc + EuroSAT-C mCE + ECE/NLL + cross-dataset acc/ECE.

**Colab:** 64x64 27k imgs, ResNet-18 ~20-40min T4; corruption + cross-dataset eval inference-only. Total 3-5h.

**Hypothesis:** AugMix + DualFocal cuts EuroSAT-C ECE >=5pp and mCE >=8pp vs CE at <=0.5pp clean cost, improves cross-dataset acc >=3pp. Negative transfer still publishable.

---

## #3 — Watts per Calibration Point (Green-AI)
**Title:** Watts per Calibration Point: Energy-ECE Pareto of Focal, Dual-Focal and Temperature Scaling Under Corruption Shift

**Gap:**
1. TMLR 2025 shift-calibration survey — exhaustive ECE under shift, 0 Joules.
2. Khan 2025 Green-AI review + Rojahn Nov 2025 meta (93 items) — inconsistent energy reporting, no Pareto standard.
3. ML.ENERGY NeurIPS25 Spotlight + Fischer Sep 2025 Ground-Truthing — rigorous energy for LLM/diffusion inference or generic training (errors up to 40%, dynamic -20 to -30%), never for calibration-loss choice on T4.

**Dataset + size:** CIFAR-10 python 170MB (50k/10k 32x32) + CIFAR-10-C tar 2.9GB, use 4 corruptions x 5 severities ~700MB streamed. Total <2GB. Sources: cs.toronto.edu/~kriz/cifar.html, Zenodo 2535967.

**Baselines:** CE, Label-Smoothing 0.05, Focal g=3 + FLSD-53, Dual-Focal g=5, + TS/ETS/Focal-TS. Metrics ECE15, AdaECE, NLL, Brier, acc; CodeCarbon + NVML Joules. Repos: `Linwei94/DualFocalLoss`, `kuangliu/pytorch-cifar`, `mlco2/codecarbon`.

**Colab:** MobileNetV2/ResNet-18 batch 128 SGD 100 epochs ~35-55min T4; TS fit CPU seconds; CIFAR-C eval CPU. Total ~4-5h splittable.

**Hypothesis:** Dual-Focal cuts mean ECE sev 3-5 >=25% vs CE+TS at <=10% extra Joules; CE+TS dominates low-energy end, FCL/Dual-Focal low-ECE end; params/FLOPs rank != Joules rank.

---

## #4 — Calibrated Tabular Ensembles
**Title:** Calibrated Parameter-Efficient Ensembles: Closing the Reliability Gap of Tabular Deep Learning Without Losing Accuracy

**Gap:**
1. Erickson et al. TabArena 2025 NeurIPS (living bench, ~25M runs, 15yr wall-clock, 51 tasks) — ranks TabM/RealMLP/ModernNCA/TabPFN-v2 vs CatBoost/LightGBM/XGBoost on acc/AUC/log-loss only; no ECE/Brier.
2. Holzmuller et al. arXiv:2407.04491 RealMLP (NeurIPS 2024) — competitive on 118+90 sets 1K-500K rows, scores/time only, no reliability diagrams.
3. Hollmann TabPFN Nature 2025 + Borisov survey IEEE TNNLS 2024 — heterogeneity/missingness/outliers as challenges; calibration absent.

**Dataset + size:** OpenML-CC18 (72 curated sets 500-96K rows, largest 5-50MB, full <2GB; use 15-20 smallest) + 5 CTR23 regression sets for Brier check. Total downloaded <1GB subsampled.

**Baselines:** CatBoost, LightGBM, TabM (`pip install tabm`, k=32), RealMLP (`pytabkit`), FT-Transformer. Post-hoc TS + isotonic (sklearn). Metrics Acc/AUC + ECE-15 + NLL + Brier. Ablate k={1,8,32}.

**Colab:** TabM-mini + RealMLP defaults 2-8min/set T4, <15min CPU; isotonic CPU seconds. 20 sets x 5 methods ~3-5h checkpointed per-dataset.

**Hypothesis:** Uncalibrated TabM/FT-T have >=2x ECE of CatBoost at matched acc on >=60% CC18 sets; isotonic cuts ECE >=40% with <=0.5pp acc loss, making calibrated TabM best acc-ECE Pareto.

---

## Pick Guide
- Safest/cheapest: #1 (4-6h, pip+API, tiny imgs).
- Highest novelty/domain: #2 (no EuroSAT-C exists).
- Green-AI angle: #3 (first Pareto plot).
- CPU-friendly: #4 (no foundation GPU).
