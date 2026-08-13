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

## Documentation (Sphinx)

Online: [Sphinx docs (GitHub Pages)](https://thefalcon1977.github.io/dental-radiography-classification/)

Build locally:

```bash
pip install -r docs/requirements.txt
cd docs && make html
# open docs/_build/html/index.html
```

## Project Layout

```
densNet/
├── main.py                       # Unified CLI (--train / --predict / …)
├── densnet/                      # Shared library (device, model, predict, …)
├── tests/                        # pytest suite
├── requirements.txt
├── requirements-dev.txt          # Commitizen + pre-commit
├── pyproject.toml                # Project metadata + Commitizen
├── .pre-commit-config.yaml       # Git hooks
├── docs/                         # Sphinx documentation
├── training_history.png          # Train/val loss & accuracy curves
├── confusion_matrix.png          # Test-set confusion matrix
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
python main.py --train
```

Saves the best checkpoint and writes training/confusion plots:

![Training history](training_history.png)

![Confusion matrix — test set](confusion_matrix.png)

## Detection (Full Image)

Interactive sliding-window scan with colored boxes:

- Red = dentin · Green = enamel · Blue = pulp

```bash
python main.py --detect
# or: python main.py --detect --image path/to/xray.png
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

Classify all images in the matching `image-testing/*_test` folder and write a CSV under `test_predictions/`.

CSV columns (same shape as external evaluation exports):

`file,class_name,target,target_label,pred_idx,pred_label,prob_positive`

`prob_positive` is the probability of the target class (P(dentin), P(enamel), or P(pulp)).

```bash
python main.py --predict all
# or one class: dentin | enamel | pulp
python main.py --predict dentin
```

Outputs:

| Command | Input | Output |
|---------|-------|--------|
| `--predict dentin` | `image-testing/dentin_test/` | `test_predictions/dentin_test_predictions.csv` |
| `--predict enamel` | `image-testing/enamel_test/` | `test_predictions/enamel_test_predictions.csv` |
| `--predict pulp` | `image-testing/pulp_test/` | `test_predictions/pulp_test_predictions.csv` |

## Evaluation Metrics

After the three prediction CSVs exist:

```bash
python main.py --evaluate
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
python main.py --help
python main.py --train
python main.py --predict all
python main.py --evaluate
python main.py --detect --image path/to/xray.png
```

## Notes

- Device is chosen automatically: CUDA → MPS → CPU
- Predictions expect images as `.png` / `.jpg` / `.jpeg`
- Re-run `--predict` before `--evaluate` if you change the model checkpoint

## Development

Install Commitizen (`cz commit`), pre-commit, and Ruff, then enable Git hooks:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

`pre-commit install` registers both the `pre-commit` and `commit-msg` hooks (see `default_install_hook_types` in `.pre-commit-config.yaml`).

Write [Conventional Commits](https://www.conventionalcommits.org/) with the Commitizen prompt:

```bash
cz commit
```

Hooks then lint staged files and reject commit messages that are not conventional (`feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`, …).

Lint and format Python with Ruff (also runs via pre-commit):

```bash
ruff check .
ruff format
```

Run the test suite:

```bash
pytest
```

Run all hooks on the tree:

```bash
pre-commit run --all-files
```

Bump the version and changelog from conventional commit history:

```bash
cz bump
```

## License

MIT
