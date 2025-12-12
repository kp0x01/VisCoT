#!/usr/bin/env python3
import argparse
import csv
import json
import os
from collections import Counter


def load_groundtruth_from_csv(csv_path):
    """
    Load ground truth from a CSV with:
      - 2nd column: image path (e.g., /content/temporal_concat/0.jpg)
      - 3rd column: ground truth label
    The file may or may not have a header row.

    Returns: dict[image_name] = label
    """
    gt = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        # Peek at first row to detect header
        first = next(reader, None)
        if first is None:
            raise ValueError("CSV is empty")

        def looks_like_header(row):
            # very simple heuristic for your case
            joined = " ".join(c.lower() for c in row)
            return (
                "image_path" in joined
                or "ground" in joined
                or "index" in joined
            )

        if looks_like_header(first):
            # skip header; everything else is data
            pass
        else:
            # first row is data, process it
            if len(first) < 3:
                raise ValueError(f"Row has fewer than 3 columns: {first}")
            image_path = first[1].strip()
            label = first[2].strip()
            image_name = os.path.basename(image_path)
            gt[image_name] = label

        # process remaining rows
        for row in reader:
            if not row:
                continue
            if len(row) < 3:
                raise ValueError(f"Row has fewer than 3 columns: {row}")
            image_path = row[1].strip()
            label = row[2].strip()
            image_name = os.path.basename(image_path)
            if image_name in gt and gt[image_name] != label:
                print(
                    f"Warning: duplicate ground truth for {image_name}, "
                    f"old='{gt[image_name]}', new='{label}'. Keeping first."
                )
                continue
            gt[image_name] = label

    return gt


def load_predictions_json(json_path, id_key="image_name", pred_key="prediction"):
    """
    Load predictions from your JSON:
      [
        {
          "image_name": "0.jpg",
          "image_path": "...",
          "prediction": "first",
          ...
        },
        ...
      ]
    Returns: dict[image_name] = prediction
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Expected JSON to be a list of records.")

    preds = {}
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            raise ValueError(f"Element {i} is not a dict: {rec}")
        if id_key not in rec or pred_key not in rec:
            raise ValueError(
                f"Element {i} missing '{id_key}' or '{pred_key}'. "
                f"Keys present: {list(rec.keys())}"
            )
        image_name = str(rec[id_key]).strip()
        pred = str(rec[pred_key]).strip()
        preds[image_name] = pred

    return preds


def compute_metrics(preds, gt):
    """
    Compute:
      - overall accuracy
      - per-label accuracy, precision, recall, f1
      - confusion matrix
    Only over IDs present in BOTH preds and gt.
    """
    y_true = []
    y_pred = []

    missing_in_gt = []
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

    # Per-label accuracy
    label_counts = Counter(y_true)
    label_correct = Counter()
    for p, t in zip(y_pred, y_true):
        if p == t:
            label_correct[t] += 1
    per_label_acc = {
        label: label_correct[label] / count for label, count in label_counts.items()
    }

    # Confusion matrix
    labels = sorted(set(y_true) | set(y_pred))
    confusion = {t: Counter() for t in labels}
    for p, t in zip(y_pred, y_true):
        confusion[t][p] += 1

    # Compute precision, recall, and F1 for each label
    precision = {}
    recall = {}
    f1_scores = {}
    
    for label in labels:
        # True Positives: predicted as label and actually is label
        tp = sum(1 for p, t in zip(y_pred, y_true) if p == label and t == label)
        # False Positives: predicted as label but actually isn't
        fp = sum(1 for p, t in zip(y_pred, y_true) if p == label and t != label)
        # False Negatives: not predicted as label but actually is
        fn = sum(1 for p, t in zip(y_pred, y_true) if p != label and t == label)
        
        # Precision: TP / (TP + FP)
        if tp + fp > 0:
            precision[label] = tp / (tp + fp)
        else:
            precision[label] = 0.0
            
        # Recall: TP / (TP + FN)
        if tp + fn > 0:
            recall[label] = tp / (tp + fn)
        else:
            recall[label] = 0.0
            
        # F1 Score: 2 * (precision * recall) / (precision + recall)
        if precision[label] + recall[label] > 0:
            f1_scores[label] = 2 * (precision[label] * recall[label]) / (precision[label] + recall[label])
        else:
            f1_scores[label] = 0.0

    # Compute macro-averaged metrics
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


def print_metrics(metrics):
    print("=== Overall ===")
    print(f"Total examples (matched image_names): {metrics['total']}")
    print(f"Correct predictions:                 {metrics['correct']}")
    print(f"Accuracy:                            {metrics['accuracy']:.4f}")
    print(f"Macro-averaged Precision:            {metrics['macro_precision']:.4f}")
    print(f"Macro-averaged Recall:               {metrics['macro_recall']:.4f}")
    print(f"Macro-averaged F1:                   {metrics['macro_f1']:.4f}")
    print()

    print("Per-label metrics")
    labels = sorted(set(metrics["precision"].keys()) | set(metrics["recall"].keys()))
    print(f"{'Label':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 55)
    for label in labels:
        acc = metrics["per_label_acc"].get(label, 0.0)
        prec = metrics["precision"].get(label, 0.0)
        rec = metrics["recall"].get(label, 0.0)
        f1 = metrics["f1_scores"].get(label, 0.0)
        print(f"{label:<15} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f}")
    print()

    print("Confusion matrix (rows = true, cols = pred)")
    labels = sorted(set(metrics["confusion"].keys()) | 
                   set(label for counts in metrics["confusion"].values() for label in counts.keys()))
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy, precision, recall metrics by matching JSON predictions to CSV ground truth via image filename."
    )
    parser.add_argument(
        "--pred_json",
        type=str,
        default="/workspace/Visual-CoT/temporal_ordering_results.json",
        help="Path to JSON file with predictions (has 'image_name' and 'prediction').",
    )
    parser.add_argument(
        "--gt_csv",
        type=str,
        default="/workspace/Visual-CoT/predictions_temporal_vqa_improved_prompt_2.csv",
        help="Path to CSV file with ground truth (2nd col image_path, 3rd col label).",
    )

    args = parser.parse_args()

    gt = load_groundtruth_from_csv(args.gt_csv)
    preds = load_predictions_json(args.pred_json)

    metrics = compute_metrics(preds, gt)
    print_metrics(metrics)


if __name__ == "__main__":
    main()