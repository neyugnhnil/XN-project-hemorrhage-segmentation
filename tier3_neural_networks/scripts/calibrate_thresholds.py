#!/usr/bin/env python3

import argparse
import numpy as np

from shared.config_utils import load_config
from shared.eval_utils import (apply_thresholds, macro_f1, per_label_f1, summarize_classification_metrics)
from shared.json_and_csv_utils import write_json
from shared.run_model_inference import (run_classification_inference, run_multitask_inference)

# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model-family", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--val-indices", type=str, required=True)
    parser.add_argument("--thresholds-out", type=str, required=True)
    parser.add_argument("--summary-out", type=str, required=True)
    return parser.parse_args()

# defines candidate thresholds to sweep with
def threshold_grid() -> np.ndarray:
    # 0.05 to 0.95 inclusive, step 0.01
    return np.round(np.arange(0.05, 0.951, 0.01), 2)

# calibrates threshold for one class label to maximize F1 on val
def optimal_threshold_for_label(y_true_label: np.ndarray, y_prob_label: np.ndarray, grid: np.ndarray)\
     -> tuple[float, float]:

    optimal_threshold = 0.5
    optimal_f1 = -1.0 

    for threshold in grid:
        y_pred_label = (y_prob_label >= threshold).astype(np.int32)
        f1 = per_label_f1(
            y_true=y_true_label.reshape(-1, 1),
            y_pred=y_pred_label.reshape(-1, 1),
            class_names=["label"]
            )["label"]
        if f1 > optimal_f1:
            optimal_f1 = f1
            optimal_threshold = float(threshold)

    return optimal_threshold, optimal_f1

# runs threshold sweep on all labels
def calibrate_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[np.ndarray, list[float]]:

    grid = threshold_grid()
    optimal_thresholds: list[float] = []
    optimal_label_f1s: list[float] = []

    for i in range(y_true.shape[1]):
        optimal_threshold, optimal_f1 = \
            optimal_threshold_for_label(
                y_true_label=y_true[:, i],
                y_prob_label=y_prob[:, i],
                grid=grid
            )
        optimal_thresholds.append(optimal_threshold)
        optimal_label_f1s.append(optimal_f1)

    # return array of optimal thresholds, list of optimal f1s
    return np.array(optimal_thresholds, dtype=np.float32), optimal_label_f1s

# inference on val dispatcher
def load_val_outputs(config_path: str, model_family: str, weights_path: str, val_indices_path: str, batch_size: int)\
     -> dict:

    if model_family == "multiunet":
        return run_multitask_inference(
            config_path=config_path,
            model_path=weights_path,
            split_indices_path=val_indices_path,
            batch_size=batch_size
            )

    if model_family in {"mlp", "cnn"}:
        return run_classification_inference(
            config_path=config_path,
            model_path=weights_path,
            split_indices_path=val_indices_path,
            batch_size=batch_size
            )

    raise ValueError("model_family must be one of: mlp, cnn, multiunet")


def build_thresh_calibration_summary(
    model_family: str,
    class_names: list[str],
    y_true: np.ndarray, # val truth
    y_prob: np.ndarray, # val preds (probs)
    optimal_thresholds: np.ndarray,
    optimal_label_f1s: list[float]
    ) -> dict:

    # default threshold metrics
    default_thresholds = np.full(y_prob.shape[1], 0.5, dtype=np.float32)
    y_pred_default = apply_thresholds(y_prob, default_thresholds)
    default_metrics = summarize_classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred_default,
        class_names=class_names)

    # calibrated threshold metrics
    y_pred_optimal = apply_thresholds(y_prob, optimal_thresholds)
    optimal_metrics = summarize_classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred_optimal,
        class_names=class_names)

    return {
        "model_family": model_family,
        "class_names": class_names,
        "optimal_thresholds": {
            label: float(threshold)
            for label, threshold in zip(class_names, optimal_thresholds.tolist())
        },
        "optimal_val_binary_f1_by_label_during_sweep": {
            label: float(score)
            for label, score in zip(class_names, optimal_label_f1s)
        },
        "val_metrics_default_threshold_0p5": default_metrics,
        "val_metrics_optimal_thresholds": optimal_metrics,
        "macro_f1_gain": float(
            macro_f1(y_true, y_pred_optimal) - macro_f1(y_true, y_pred_default)
        )
    }


def main() -> None:
    # parse args
    args = parse_args()

    # get config and specs
    cfg = load_config(args.config)
    class_names = list(cfg["data"]["class_names"])
    if args.model_family == "mlp":
        batch_size = int(cfg["mlp"]["batch_size"])
    elif args.model_family == "cnn":
        batch_size = int(cfg["cnn"]["batch_size"])
    elif args.model_family == "multiunet":
        batch_size = int(cfg["multiunet"]["batch_size"])
    else:
        raise ValueError("model_family must be one of: mlp, cnn, multiunet")

    # get val inference outputs
    outputs = load_val_outputs(
        config_path=args.config,
        model_family=args.model_family,
        weights_path=args.weights,
        val_indices_path=args.val_indices,
        batch_size=batch_size
        )
    y_true = outputs["y_true_cls"]
    y_prob = outputs["y_prob_cls"]

    # do threshold sweep
    optimal_thresholds, optimal_label_f1s = \
        calibrate_thresholds(y_true=y_true,y_prob=y_prob)

    # build threshold reference artifact
    thresholds_payload = {
        "model_family": args.model_family,
        "class_names": class_names,
        "optimal_thresholds": optimal_thresholds.tolist()
        }

    # write summary
    summary = build_thresh_calibration_summary(
        model_family=args.model_family,
        class_names=class_names,
        y_true=y_true,
        y_prob=y_prob,
        optimal_thresholds=optimal_thresholds,
        optimal_label_f1s=optimal_label_f1s
        )

    write_json(thresholds_payload, args.thresholds_out)
    write_json(summary, args.summary_out)

    print(f"Saved thresholds: {args.thresholds_out}")
    print(f"Saved summary: {args.summary_out}")


if __name__ == "__main__":
    main()