# Dental Radiography Classification (DenseNet)

Classify dental tissue in radiography images into **dentin**, **enamel**, and **pulp** using DenseNet121.

## What It Does

- Train a 3-class DenseNet121 classifier on segmented dental patches
- Run sliding-window detection on full radiographs
- Batch-predict held-out test folders (`dentin` / `enamel` / `pulp`)
- Compute Precision, Recall, Accuracy, and F1 from prediction CSVs

## Requirements

- Python 3.10+
- PyTorch with CUDA, Apple MPS, or CPU
- Dependencies in `requirements.txt`

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project Layout

```
densNet/
├── train.py                      # Train DenseNet121
├── detect_simple.py              # Sliding-window detection (interactive)
├── predict_dentin_test.py        # Predict dentin_test → CSV
├── predict_enamel_test.py        # Predict enamel_test → CSV
├── predict_pulp_test.py          # Predict pulp_test → CSV
├── evaluate_test_predictions.py  # Metrics from prediction CSVs
├── requirements.txt
├── slm/                          # Saved model checkpoints
├── image-testing/
│   ├── dentin_test/
│   ├── enamel_test/
│   └── pulp_test/
├── segmented_dental_adiography/  # train / valid / test for training
└── test_predictions/             # CSV outputs + evaluation report
```

## Model Checkpoint

Inference scripts use:

```text
slm/resolution_best_densenet_model.pth
```

Classes (index order): `0 = dentin`, `1 = enamel`, `2 = pulp`.

## Training

Dataset layout:

```text
segmented_dental_adiography/
├── train/{dentin,enamel,pulp}/
├── valid/{dentin,enamel,pulp}/
└── test/{dentin,enamel,pulp}/
```

```bash
python train.py
```

Saves the best checkpoint and writes training/confusion plots.

## Detection (Full Image)

Interactive sliding-window scan with colored boxes:

- Red = dentin · Green = enamel · Blue = pulp

```bash
python detect_simple.py
# Enter image path when prompted
```

## External Test Images (`image-testing/`)

Held-out tissue patches for batch evaluation (separate from `segmented_dental_adiography/` used in training).

```text
image-testing/
├── dentin_test/   # 75 images — ground truth: dentin
├── enamel_test/   # 75 images — ground truth: enamel
└── pulp_test/     # 75 images — ground truth: pulp
```

| Folder | Class | Count |
|--------|-------|-------|
| `dentin_test/` | dentin | 75 |
| `enamel_test/` | enamel | 75 |
| `pulp_test/` | pulp | 75 |
| **Total** | | **225** |

Supported formats: `.png`, `.jpg`, `.jpeg` (any casing).

## Batch Test Predictions

Each script classifies all images in the matching `image-testing/*_test` folder and writes a CSV under `test_predictions/`.

CSV columns (same shape as external evaluation exports):

`file,class_name,target,target_label,pred_idx,pred_label,prob_positive`

`prob_positive` is the probability of that script’s target class (P(dentin), P(enamel), or P(pulp)).

```bash
python predict_dentin_test.py
python predict_enamel_test.py
python predict_pulp_test.py
```

Outputs:

| Script | Input | Output |
|--------|-------|--------|
| `predict_dentin_test.py` | `image-testing/dentin_test/` | `test_predictions/dentin_test_predictions.csv` |
| `predict_enamel_test.py` | `image-testing/enamel_test/` | `test_predictions/enamel_test_predictions.csv` |
| `predict_pulp_test.py` | `image-testing/pulp_test/` | `test_predictions/pulp_test_predictions.csv` |

## Evaluation Metrics

After the three prediction CSVs exist:

```bash
python evaluate_test_predictions.py
```

Uses:

| Metric | Formula |
|--------|---------|
| Precision | TP / (TP + FP) |
| Recall | TP / (TP + FN) |
| Accuracy | (TP + TN) / (TP + TN + FP + FN) |
| F1 | 2 × Precision × Recall / (Precision + Recall) |

Writes:

- `test_predictions/evaluation_metrics.csv`
- `test_predictions/evaluation_report.txt`
- See also `test_predictions/evaluation_report.md`

### Latest External Test Results (225 images)

| Class | Precision | Recall | Accuracy | F1 |
|-------|-----------|--------|----------|-----|
| dentin | 95.38% | 82.67% | 92.89% | 88.57% |
| enamel | 94.87% | 98.67% | 97.78% | 96.73% |
| pulp | 87.80% | 96.00% | 94.22% | 91.72% |
| **macro** | **92.69%** | **92.44%** | **94.96%** | **92.34%** |

**Overall accuracy:** 92.44%

## Typical Workflow

```bash
# 1) Train (optional if checkpoint already exists)
python train.py

# 2) Batch-predict each test folder
python predict_dentin_test.py
python predict_enamel_test.py
python predict_pulp_test.py

# 3) Build metrics report
python evaluate_test_predictions.py

# 4) Optional: inspect a full radiograph
python detect_simple.py
```

## Notes

- Device is chosen automatically: CUDA → MPS → CPU
- Prediction scripts expect images as `.png` / `.jpg` / `.jpeg`
- Re-run predictions before evaluation if you change the model checkpoint

## License

MIT
