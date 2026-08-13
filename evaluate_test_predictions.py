"""
Evaluate tissue classification from test_predictions CSV files.

Computes Precision, Recall, Accuracy, and F1-score using:
  Precision = TP / (TP + FP)
  Recall    = TP / (TP + FN)
  Accuracy  = (TP + TN) / (TP + TN + FP + FN)
  F1        = 2 * Precision * Recall / (Precision + Recall)

Reads:
  test_predictions/dentin_test_predictions.csv
  test_predictions/enamel_test_predictions.csv
  test_predictions/pulp_test_predictions.csv

Writes:
  test_predictions/evaluation_metrics.csv
  test_predictions/evaluation_report.txt
"""

from __future__ import annotations

import csv
import os
from typing import TypeAlias

CLASS_NAMES: list[str] = ["dentin", "enamel", "pulp"]
PREDICTIONS_DIR = "test_predictions"
INPUT_CSVS: list[str] = [
    os.path.join(PREDICTIONS_DIR, "dentin_test_predictions.csv"),
    os.path.join(PREDICTIONS_DIR, "enamel_test_predictions.csv"),
    os.path.join(PREDICTIONS_DIR, "pulp_test_predictions.csv"),
]
OUTPUT_METRICS_CSV = os.path.join(PREDICTIONS_DIR, "evaluation_metrics.csv")
OUTPUT_REPORT_TXT = os.path.join(PREDICTIONS_DIR, "evaluation_report.txt")

IntList: TypeAlias = list[int]
CountDict: TypeAlias = dict[str, int]
MetricDict: TypeAlias = dict[str, float | int | str]
ConfusionCounts: TypeAlias = dict[str, CountDict]


def safe_div(numerator: float, denominator: float) -> float:
    """Divide safely; return 0.0 when denominator is zero.

    Args:
        numerator: Division numerator.
        denominator: Division denominator.

    Returns:
        Quotient, or 0.0 if denominator is 0.
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def load_predictions(csv_paths: list[str]) -> tuple[IntList, IntList]:
    """Load target and predicted class indices from prediction CSVs.

    Args:
        csv_paths: Paths to prediction CSV files.

    Returns:
        Tuple of (targets, predictions) as integer class indices.

    Raises:
        SystemExit: If a required CSV is missing.
    """
    targets: IntList = []
    preds: IntList = []

    for path in csv_paths:
        if not os.path.exists(path):
            raise SystemExit(f"Missing predictions file: {path}")

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                targets.append(int(row["target"]))
                preds.append(int(row["pred_idx"]))

    return targets, preds


def confusion_for_class(
    targets: IntList,
    preds: IntList,
    class_idx: int,
) -> CountDict:
    """Compute one-vs-rest TP/FP/TN/FN for a single class.

    Args:
        targets: Ground-truth class indices.
        preds: Predicted class indices.
        class_idx: Positive class index.

    Returns:
        Dict with keys TP, FP, TN, FN.
    """
    tp = fp = tn = fn = 0
    for y_true, y_pred in zip(targets, preds):
        true_pos = y_true == class_idx
        pred_pos = y_pred == class_idx
        if true_pos and pred_pos:
            tp += 1
        elif (not true_pos) and pred_pos:
            fp += 1
        elif true_pos and (not pred_pos):
            fn += 1
        else:
            tn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def metrics_from_counts(counts: CountDict) -> MetricDict:
    """Compute Precision, Recall, Accuracy, F1 from confusion counts.

    Args:
        counts: Dict with TP, FP, TN, FN.

    Returns:
        Metric dict including counts and derived scores.
    """
    tp = counts["TP"]
    fp = counts["FP"]
    tn = counts["TN"]
    fn = counts["FN"]

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }


def overall_accuracy(targets: IntList, preds: IntList) -> float:
    """Compute multi-class accuracy = correct / total.

    Args:
        targets: Ground-truth class indices.
        preds: Predicted class indices.

    Returns:
        Overall accuracy in [0, 1].
    """
    if not targets:
        return 0.0
    correct = sum(1 for y_true, y_pred in zip(targets, preds) if y_true == y_pred)
    return correct / len(targets)


def confusion_matrix(
    targets: IntList,
    preds: IntList,
    num_classes: int,
) -> list[list[int]]:
    """Build a square confusion matrix [true][pred].

    Args:
        targets: Ground-truth class indices.
        preds: Predicted class indices.
        num_classes: Number of classes.

    Returns:
        Nested list matrix of shape (num_classes, num_classes).
    """
    matrix = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for y_true, y_pred in zip(targets, preds):
        matrix[y_true][y_pred] += 1
    return matrix


def format_pct(value: float) -> str:
    """Format a ratio as a percentage string.

    Args:
        value: Ratio in [0, 1].

    Returns:
        Percentage string with 2 decimal places.
    """
    return f"{value * 100:.2f}%"


def build_report(
    class_metrics: list[MetricDict],
    macro: MetricDict,
    overall_acc: float,
    matrix: list[list[int]],
    n_samples: int,
) -> str:
    """Build a human-readable evaluation report.

    Args:
        class_metrics: Per-class metric dicts (with class_name).
        macro: Macro-averaged metrics.
        overall_acc: Overall multi-class accuracy.
        matrix: Confusion matrix.
        n_samples: Total number of samples.

    Returns:
        Report text.
    """
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("DenseNet Tissue Classification — Evaluation Report")
    lines.append("=" * 72)
    lines.append(f"Total samples: {n_samples}")
    lines.append(f"Overall Accuracy: {format_pct(overall_acc)}")
    lines.append("")
    lines.append("Formulas:")
    lines.append("  Precision = TP / (TP + FP)")
    lines.append("  Recall    = TP / (TP + FN)")
    lines.append("  Accuracy  = (TP + TN) / (TP + TN + FP + FN)")
    lines.append("  F1        = 2 * Precision * Recall / (Precision + Recall)")
    lines.append("")
    lines.append("-" * 72)
    lines.append(
        f"{'Class':<10} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} "
        f"{'Prec':>8} {'Recall':>8} {'Acc':>8} {'F1':>8}"
    )
    lines.append("-" * 72)

    for m in class_metrics:
        lines.append(
            f"{str(m['class_name']):<10} "
            f"{int(m['TP']):>4} {int(m['FP']):>4} {int(m['TN']):>4} {int(m['FN']):>4} "
            f"{format_pct(float(m['precision'])):>8} "
            f"{format_pct(float(m['recall'])):>8} "
            f"{format_pct(float(m['accuracy'])):>8} "
            f"{format_pct(float(m['f1'])):>8}"
        )

    lines.append("-" * 72)
    lines.append(
        f"{'macro':<10} "
        f"{'':>4} {'':>4} {'':>4} {'':>4} "
        f"{format_pct(float(macro['precision'])):>8} "
        f"{format_pct(float(macro['recall'])):>8} "
        f"{format_pct(float(macro['accuracy'])):>8} "
        f"{format_pct(float(macro['f1'])):>8}"
    )
    lines.append("")
    lines.append("Confusion Matrix (rows=true, cols=pred):")
    header = "true\\pred".ljust(10) + "".join(f"{c:>10}" for c in CLASS_NAMES)
    lines.append(header)
    for i, row in enumerate(matrix):
        lines.append(CLASS_NAMES[i].ljust(10) + "".join(f"{v:>10}" for v in row))
    lines.append("=" * 72)
    return "\n".join(lines) + "\n"


def main() -> None:
    """Evaluate predictions and write metrics CSV + text report."""
    targets, preds = load_predictions(INPUT_CSVS)
    n_samples = len(targets)
    if n_samples == 0:
        raise SystemExit("No prediction rows found.")

    class_metrics: list[MetricDict] = []
    for idx, name in enumerate(CLASS_NAMES):
        counts = confusion_for_class(targets, preds, idx)
        metrics = metrics_from_counts(counts)
        metrics["class_name"] = name
        class_metrics.append(metrics)

    macro: MetricDict = {
        "class_name": "macro",
        "TP": "",
        "FP": "",
        "TN": "",
        "FN": "",
        "precision": sum(float(m["precision"]) for m in class_metrics) / len(CLASS_NAMES),
        "recall": sum(float(m["recall"]) for m in class_metrics) / len(CLASS_NAMES),
        "accuracy": sum(float(m["accuracy"]) for m in class_metrics) / len(CLASS_NAMES),
        "f1": sum(float(m["f1"]) for m in class_metrics) / len(CLASS_NAMES),
    }

    overall_acc = overall_accuracy(targets, preds)
    matrix = confusion_matrix(targets, preds, len(CLASS_NAMES))

    fieldnames = [
        "class_name",
        "TP",
        "FP",
        "TN",
        "FN",
        "precision",
        "recall",
        "accuracy",
        "f1",
    ]
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    with open(OUTPUT_METRICS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in class_metrics:
            writer.writerow({k: row[k] for k in fieldnames})
        writer.writerow({k: macro[k] for k in fieldnames})
        writer.writerow(
            {
                "class_name": "overall_accuracy",
                "TP": "",
                "FP": "",
                "TN": "",
                "FN": "",
                "precision": "",
                "recall": "",
                "accuracy": overall_acc,
                "f1": "",
            }
        )

    report = build_report(class_metrics, macro, overall_acc, matrix, n_samples)
    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"Wrote metrics -> {OUTPUT_METRICS_CSV}")
    print(f"Wrote report  -> {OUTPUT_REPORT_TXT}")


if __name__ == "__main__":
    main()
