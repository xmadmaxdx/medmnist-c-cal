# Calibrated Compact CNNs Under Clinical Corruptions (MedMNIST-C-Cal)

Winner #1 from `TitleResearchResult.md`: joint robustness-calibration of lightweight
CNNs on BloodMNIST-C + DermaMNIST-C with calibration-aware losses.

Novelty: MedMNIST-C measured accuracy-only (BE/rBE). This repo adds ECE/NLL/Brier,
temperature scaling, reliability diagrams, and severity curves for CE vs Focal vs DualFocal.

## Datasets (Colab only, <100MB total)
- BloodMNIST 17,092 images, 8 classes, 28x28 RGB, 35.5MB npz
- DermaMNIST 10,015 images, 7 classes, 28x28 RGB, 19.7MB npz
- Total <100MB, well under 20GB cap. CC BY 4.0 (Derma CC BY-NC 4.0).

## Run in Colab — complete copy-paste cells in order

Cell 0 — YOUR WORK: open fresh T4 GPU runtime, paste to check it.
WHAT HAPPENS: prints GPU + python; if No GPU, use Runtime > Change runtime type > T4.
```bash
!nvidia-smi --query-gpu=name,memory.total --format=csv
!python --version
```

Cell 1 — YOUR WORK: paste as-is (URL already set).
WHAT HAPPENS: clones project, enters folder.
```bash
!git clone https://github.com/xmadmaxdx/medmnist-c-cal.git medmnist-c-cal
%cd medmnist-c-cal
!pwd
!ls
```

Cell 1b — YOUR WORK (do FIRST before training): connect Drive for autosave.
WHAT HAPPENS: mounts `/content/drive`, creates `MyDrive/medmnist-c-cal-outputs/`. Every `best.pt` + `last.pt` + `history.json` + results auto-mirrors there each save. Local runs skip silently.
```bash
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
!mkdir -p /content/drive/MyDrive/medmnist-c-cal-outputs
!ls -ld /content/drive/MyDrive/medmnist-c-cal-outputs
```

Cell 2 — YOUR WORK: paste once, wait ~3-6 min for pip.
WHAT HAPPENS: installs torch + medmnist + sklearn + matplotlib via `colab_setup.sh`.
```bash
!bash colab_setup.sh
```

Cell 3 — YOUR WORK (recommended): paste to prove code works before download.
WHAT HAPPENS: 1-epoch fake-data train + eval on CPU, ~1-2 min. Creates `outputs/checkpoints-smoke/` + `outputs/smoke-results.json`.
```bash
!python -m pytest tests/test_smoke.py -v
!python -m scripts.train --config configs/smoke.yaml --smoke
!python -m scripts.evaluate --config configs/smoke.yaml --smoke --out outputs/smoke-results.json
```

Cell 4 — YOUR WORK: paste to download real data. Do once.
WHAT HAPPENS: downloads BloodMNIST 35.5MB + DermaMNIST 19.7MB npz to `data/`. Total <100MB. Skips if present.
```bash
!python -m scripts.download_data --root data
!ls -lh data/
```

Cell 5 — YOUR WORK: paste, wait ~5-10 min T4 (CPU ~25 min).
WHAT HAPPENS: trains ResNet-18 DualFocal on BloodMNIST 20 epochs. Saves `outputs/checkpoints-blood/best.pt + last.pt + history.json`. Resume-safe on disconnect.
```bash
!python -m scripts.train --config configs/blood.yaml --download
```

Cell 6 — YOUR WORK: paste, wait ~3-6 min.
WHAT HAPPENS: fits temperature on val, evals clean + 8 corruptions x 5 severities. Writes `outputs/blood-results.json` + `outputs/reliability.png` + `outputs/severity_ece.png`.
```bash
!python -m scripts.evaluate --config configs/blood.yaml --download --out outputs/blood-results.json
!ls -lh outputs/
```

Cell 7 — YOUR WORK: paste, wait ~4-8 min T4.
WHAT HAPPENS: same for DermaMNIST 7 classes.
```bash
!python -m scripts.train --config configs/derma.yaml --download
!python -m scripts.evaluate --config configs/derma.yaml --download --out outputs/derma-results.json
!ls -lh outputs/
```

Cell 8 — YOUR WORK: paste to see performance. No training.
WHAT HAPPENS: prints clean acc/ECE/NLL/Brier + corrupted summary.
```bash
!python -c "import json; d=json.load(open('outputs/blood-results.json')); print('BLOOD clean:', d['clean']); print('temp:', d['temperature'])"
!python -c "import json; d=json.load(open('outputs/derma-results.json')); print('DERMA clean:', d['clean']); print('temp:', d['temperature'])"
```

Cell 9 — OPTIONAL full matrix. YOUR WORK: paste only if you have ~1-2h T4.
WHAT HAPPENS: 3 losses x 2 models = 6 cells, each checkpointed to `outputs/ckpt-<dataset>-<model>-<loss>/`. Continues on single-cell fail.
```bash
!python -m scripts.run_matrix --dataset bloodmnist --epochs 20 --download
```

Cell 10 — verify Drive autosave. YOUR WORK: paste, no training.
WHAT HAPPENS: lists local + Drive mirrors of every .pt.
```bash
!ls -lh outputs/*/best.pt outputs/*/last.pt outputs/*.json 2>/dev/null; echo "--- DRIVE ---"
!find /content/drive/MyDrive/medmnist-c-cal-outputs -name "*.pt" -o -name "*.json" | sort
```

Cell 11 — OPTIONAL save. YOUR WORK: paste to keep full outputs after runtime recycles.
WHAT HAPPENS: copies `outputs/` to Drive (autosave already did .pt/.json/.png per-save; this is full backup).
```bash
!cp -r outputs /content/drive/MyDrive/medmnist-c-cal-outputs-full
!ls -lh /content/drive/MyDrive/medmnist-c-cal-outputs/
```

If disconnect: re-run Cells 1-2, then re-run train command — it resumes from `last.pt`.


## Local smoke (no download, CPU, <2 min)
```bash
pip install -r requirements.txt
python -m pytest tests/test_smoke.py -v
python -m scripts.train --config configs/smoke.yaml --smoke
python -m scripts.evaluate --config configs/smoke.yaml --smoke --out outputs/smoke-results.json
```

## Layout
- `configs/`: smoke/blood/derma YAML
- `src/data/`: MedMNIST loader, 8 clinical corruptions x severity 1-5
- `src/models/`: resnet18_small, mobilenetv3_small, simplecnn
- `src/losses/`: ce, focal, dualfocal (stable variant)
- `src/calibration/`: ECE-15, NLL, Brier, temperature scaling
- `src/engine/`: trainer with resume, clean/corrupted evaluator
- `src/viz/`: reliability + severity ECE plots
- `scripts/`: train, evaluate, run_matrix, download_data
- `tests/`: offline fake-data smoke

## Hypothesis
DualFocal cuts corrupted ECE >=4pp vs CE while targeted-style aug holds accuracy.
Fail if ECE improves ID-only. Report clean acc + corrupted acc + ECE + NLL + 3 seeds.
