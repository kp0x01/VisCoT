#!/usr/bin/env python3
"""Compare JSON predictions against the CSV ground truth for temporal ordering."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List


def load_groundtruth_from_csv(csv_path: str) -> Dict[str, str]:
    gt: Dict[str, str] = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        first = next(reader, None)
        if first is None:
            raise ValueError("CSV is empty")

        def looks_like_header(row: List[str]) -> bool:
            joined = " ".join(cell.lower() for cell in row)
            return "image_path" in joined or "ground" in joined or "index" in joined

        if not looks_like_header(first):
            if len(first) < 3:
                raise ValueError(f"Row has fewer than 3 columns: {first}")
            image_name = os.path.basename(first[1].strip())
            gt[image_name] = first[2].strip()

        for row in reader:
            if not row:
                continue
            if len(row) < 3:
                raise ValueError(f"Row has fewer than 3 columns: {row}")
            image_name = os.path.basename(row[1].strip())
            label = row[2].strip()
            if image_name in gt and gt[image_name] != label:
                print(
                    f"Warning: duplicate ground truth for {image_name}, "
                    f"old='{gt[image_name]}', new='{label}'. Keeping first."
                )
                continue
            gt[image_name] = label
    return gt


def load_predictions_json(json_path: str, id_key: str = "image_name", pred_key: str = "prediction") -> Dict[str, str]:
    with open(json_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected JSON to be a list of records.")

    preds: Dict[str, str] = {}
    for idx, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(f"Element {idx} is not a dict: {rec}")
        if id_key not in rec or pred_key not in rec:
            raise ValueError(f"Element {idx} missing '{id_key}' or '{pred_key}'. Keys present: {list(rec.keys())}")
        image_name = str(rec[id_key]).strip()
        pred = str(rec[pred_key]).strip()
        preds[image_name] = pred
    return preds


def compute_metrics(preds: Dict[str, str], gt: Dict[str, str]) -> Dict[str, object]:
    y_true: List[str] = []
    y_pred: List[str] = []
    missing_in_gt: List[str] = []

    for image_name, pred in preds.items():
        if image_name not in gt:
            missing_in_gt.append(image_name)
            continue
        y_pred.append(pred)
        y_true.append(gt[image_name])

    missing_in_pred = [name for name in gt.keys() if name not in preds]
    if not y_true:
        raise ValueError("No overlapping image names between predictions and ground truth.")

    total = len(y_true)
    correct = sum(1 for p, t in zip(y_pred, y_true) if p == t)
    accuracy = correct / total

    label_counts = Counter(y_true)
    label_correct = Counter()
    for p, t in zip(y_pred, y_true):
        if p == t:
            label_correct[t] += 1
    per_label_acc = {label: label_correct[label] / count for label, count in label_counts.items()}

    labels = sorted(set(y_true) | set(y_pred))
    confusion = {t: Counter() for t in labels}
    for p, t in zip(y_pred, y_true):
        confusion[t][p] += 1

    precision: Dict[str, float] = {}
    recall: Dict[str, float] = {}
    f1_scores: Dict[str, float] = {}
    for label in labels:
        tp = sum(1 for p, t in zip(y_pred, y_true) if p == label and t == label)
        fp = sum(1 for p, t in zip(y_pred, y_true) if p == label and t != label)
        fn = sum(1 for p, t in zip(y_pred, y_true) if p != label and t == label)
        precision[label] = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall[label] = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision[label] + recall[label] > 0:
            f1_scores[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label])
        else:
            f1_scores[label] = 0.0

    macro_precision = sum(precision.values()) / len(precision) if precision else 0
    macro_recall = sum(recall.values()) / len(recall) if recall else 0
    macro_f1 = sum(f1_scores.values()) / len(f1_scores) if f1_scores else 0

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "per_label_acc": per_label_acc,
        "precision": precision,
        "recall": recall,
        "f1_scores": f1_scores,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "confusion": confusion,
        "missing_in_gt": missing_in_gt,
        "missing_in_pred": missing_in_pred,
    }


def print_metrics(metrics: Dict[str, object]) -> None:
    print("=== Overall ===")
    print(f"Total examples (matched image_names): {metrics['total']}")
    print(f"Correct predictions:                 {metrics['correct']}")
    print(f"Accuracy:                            {metrics['accuracy']:.4f}")
    print(f"Macro-averaged Precision:            {metrics['macro_precision']:.4f}")
    print(f"Macro-averaged Recall:               {metrics['macro_recall']:.4f}")
    print(f"Macro-averaged F1:                   {metrics['macro_f1']:.4f}\n")

    print("Per-label metrics")
    labels = sorted(set(metrics["precision"].keys()) | set(metrics["recall"].keys()))
    print(f"{'Label':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 55)
    for label in labels:
        acc = metrics["per_label_acc"].get(label, 0.0)
        prec = metrics["precision"].get(label, 0.0)
        rec = metrics["recall"].get(label, 0.0)
        f1_val = metrics["f1_scores"].get(label, 0.0)
        print(f"{label:<15} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1_val:<10.4f}")
    print()

    print("Confusion matrix (rows = true, cols = pred)")
    labels = sorted(set(metrics["confusion"].keys()) | set(label for counts in metrics["confusion"].values() for label in counts.keys()))
    header = ["true \\ pred"] + list(labels)
    print("\t".join(header))
    for t in labels:
        row = [t]
        for p in labels:
            count = metrics["confusion"].get(t, {}).get(p, 0)
            row.append(str(count))
        print("\t".join(row))
    print()

    if metrics["missing_in_gt"]:
        print(f"Image names in predictions but not in ground truth: {len(metrics['missing_in_gt'])}")
    if metrics["missing_in_pred"]:
        print(f"Image names in ground truth but not in predictions: {len(metrics['missing_in_pred'])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute accuracy/precision/recall by matching JSON predictions to CSV ground truth via filename.")
    parser.add_argument(
        "--pred_json",
        type=str,
        default="/workspace/Visual-CoT/temporal_ordering_results.json",
        help="Path to JSON file with predictions (needs 'image_name' and 'prediction').",
    )
    parser.add_argument(
        "--gt_csv",
        type=str,
        default="/workspace/Visual-CoT/predictions_temporal_vqa_improved_prompt_2.csv",
        help="Path to CSV ground truth (2nd column image_path, 3rd column label).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gt = load_groundtruth_from_csv(args.gt_csv)
    preds = load_predictions_json(args.pred_json)
    metrics = compute_metrics(preds, gt)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
