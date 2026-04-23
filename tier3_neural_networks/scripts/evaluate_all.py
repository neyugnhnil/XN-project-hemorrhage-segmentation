#!/usr/bin/env python3

import argparse

import numpy as np
import pandas as pd

from shared.config_utils import get_model_batch_size, load_config
from shared.eval_utils import (apply_thresholds, mean_dice_score, summarize_classification_metrics)

from shared.json_and_csv_utils import read_json, write_dataframe, write_json
from shared.run_model_inference import (
    run_attentioncnn_inference,
    run_classification_inference,
    run_multitask_inference,
)

# CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--test-indices", type=str, required=True)

    parser.add_argument("--mlp-weights", type=str, required=True)
    parser.add_argument("--mlp-thresholds", type=str, required=True)

    parser.add_argument("--cnn-weights", type=str, required=True)
    parser.add_argument("--cnn-thresholds", type=str, required=True)

    parser.add_argument("--multiunet-weights", type=str, required=True)
    parser.add_argument("--multiunet-thresholds", type=str, required=True)

    parser.add_argument("--attentioncnn-weights", type=str, default=None)
    parser.add_argument("--attentioncnn-segmenter-weights", type=str, default=None)
    parser.add_argument("--attentioncnn-thresholds", type=str, default=None)

    parser.add_argument("--metrics-out", type=str, required=True)
    parser.add_argument("--predictions-out", type=str, required=True)
    parser.add_argument("--segmentation-out", type=str, required=True)
    parser.add_argument("--summary-out", type=str, required=True)

    return parser.parse_args()


def should_evaluate_attentioncnn(args: argparse.Namespace) -> bool:
    attention_args = [
        args.attentioncnn_weights,
        args.attentioncnn_segmenter_weights,
        args.attentioncnn_thresholds
        ]
    if any(value is not None for value in attention_args) and not all(attention_args):
        raise ValueError(
            "AttentionCNN evaluation requires --attentioncnn-weights, "
            "--attentioncnn-segmenter-weights, and --attentioncnn-thresholds"
            )
    return all(attention_args)

# get thresholds determined by the calibration step
def load_thresholds(thresholds_path: str) -> np.ndarray:
    payload = read_json(thresholds_path)

    if "optimal_thresholds" in payload:
        thresholds = payload["optimal_thresholds"]
    elif "thresholds" in payload:
        thresholds = payload["thresholds"]
    else:
        raise ValueError(f"Threshold file missing threshold values: {thresholds_path}")

    return np.asarray(thresholds, dtype=np.float32)

# turns one model’s metrics dictionary into one table row
def build_metrics_row(model_family: str, metrics: dict) -> dict:
    row = {
        "model_family": model_family,
        "macro_auc": float(metrics["macro_auc"]),
        "macro_auprc": float(metrics["macro_auprc"]),
        "macro_f1": float(metrics["macro_f1"])
        }

    for metric_name in ["per_label_auc", "per_label_auprc", "per_label_f1"]:
        metric_dict = metrics[metric_name]
        for label, value in metric_dict.items():
            row[f"{metric_name}__{label}"] = None if value is None else float(value)

    return row

# builds per-case predictions table for one model
def build_prediction_rows(
    model_family: str, 
    metadata: list[dict], 
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: list[str]
    ) -> list[dict]:
    
    rows: list[dict] = []

    for i, meta_row in enumerate(metadata):
        row = {
            "model_family": model_family,
            "meta_index": int(meta_row["meta_index"]),
            "id": meta_row["id"],
            "render_directory": meta_row["render_directory"],
            "seg_label_source": meta_row["seg_label_source"]
            }

        for j, label in enumerate(class_names):
            row[f"true__{label}"] = int(y_true[i, j])
            row[f"prob__{label}"] = float(y_prob[i, j])
            row[f"pred__{label}"] = int(y_pred[i, j])

        rows.append(row)

    return rows

# evaluation path for mlp/cnn
def evaluate_classification_model(
    model_family: str,
    config_path: str,
    weights_path: str,
    thresholds_path: str,
    test_indices_path: str,
    batch_size: int,
    class_names: list[str]) -> tuple[dict, list[dict]]:
    
    thresholds = load_thresholds(thresholds_path)
    outputs = run_classification_inference(
        config_path=config_path,
        model_path=weights_path,
        split_indices_path=test_indices_path,
        batch_size=batch_size)

    y_true = outputs["y_true_cls"]
    y_prob = outputs["y_prob_cls"]
    y_pred = apply_thresholds(y_prob, thresholds)

    metrics = summarize_classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        class_names=class_names
        )

    metrics_row = build_metrics_row(model_family, metrics)
    prediction_rows = build_prediction_rows(
        model_family=model_family,
        metadata=outputs["metadata"],
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        class_names=class_names
        )

    return metrics_row, prediction_rows

# evaluation path for multiunet
def evaluate_multiunet_model(
    config_path: str,
    weights_path: str,
    thresholds_path: str,
    test_indices_path: str,
    batch_size: int,
    class_names: list[str]) -> tuple[dict, dict, list[dict]]:

    thresholds = load_thresholds(thresholds_path)

    outputs = run_multitask_inference(
        config_path=config_path,
        model_path=weights_path,
        split_indices_path=test_indices_path,
        batch_size=batch_size)

    y_true_cls = outputs["y_true_cls"]
    y_prob_cls = outputs["y_prob_cls"]
    y_pred_cls = apply_thresholds(y_prob_cls, thresholds)

    cls_metrics = summarize_classification_metrics(
        y_true=y_true_cls,
        y_prob=y_prob_cls,
        y_pred=y_pred_cls,
        class_names=class_names)

    cls_metrics_row = build_metrics_row("multiunet", cls_metrics)

    seg_metrics_row = {
        "model_family": "multiunet",
        "mean_dice_threshold_0p5": float(
            mean_dice_score(
                outputs["y_true_seg"],
                outputs["y_prob_seg"],
                threshold=0.5,
                eps=1.0e-7)
                )
            }

    prediction_rows = build_prediction_rows(
        model_family="multiunet",
        metadata=outputs["metadata"],
        y_true=y_true_cls,
        y_prob=y_prob_cls,
        y_pred=y_pred_cls,
        class_names=class_names
        )

    return cls_metrics_row, seg_metrics_row, prediction_rows


def evaluate_attentioncnn_model(
    config_path: str,
    classifier_weights_path: str,
    segmenter_weights_path: str,
    thresholds_path: str,
    test_indices_path: str,
    batch_size: int,
    class_names: list[str]) -> tuple[dict, dict, list[dict]]:

    thresholds = load_thresholds(thresholds_path)

    outputs = run_attentioncnn_inference(
        config_path=config_path,
        classifier_model_path=classifier_weights_path,
        segmenter_model_path=segmenter_weights_path,
        split_indices_path=test_indices_path,
        batch_size=batch_size
        )

    y_true_cls = outputs["y_true_cls"]
    y_prob_cls = outputs["y_prob_cls"]
    y_pred_cls = apply_thresholds(y_prob_cls, thresholds)

    cls_metrics = summarize_classification_metrics(
        y_true=y_true_cls,
        y_prob=y_prob_cls,
        y_pred=y_pred_cls,
        class_names=class_names
        )

    cls_metrics_row = build_metrics_row("attentioncnn", cls_metrics)

    seg_metrics_row = {
        "model_family": "attentioncnn",
        "mean_dice_threshold_0p5": float(
            mean_dice_score(
                outputs["y_true_seg"],
                outputs["y_prob_seg"],
                threshold=0.5,
                eps=1.0e-7
                )
            )
        }

    prediction_rows = build_prediction_rows(
        model_family="attentioncnn",
        metadata=outputs["metadata"],
        y_true=y_true_cls,
        y_prob=y_prob_cls,
        y_pred=y_pred_cls,
        class_names=class_names
        )

    return cls_metrics_row, seg_metrics_row, prediction_rows


def main() -> None:
    # parse args
    args = parse_args()

    # load config and specs
    cfg = load_config(args.config)
    class_names = list(cfg["data"]["class_names"])

    # initialize results containers
    metrics_rows: list[dict] = []
    segmentation_rows: list[dict] = []
    prediction_rows: list[dict] = []
    models_evaluated = ["mlp", "cnn", "multiunet"]

    # evaluate mlp
    mlp_metrics_row, mlp_prediction_rows =\
    evaluate_classification_model(
        model_family="mlp",
        config_path=args.config,
        weights_path=args.mlp_weights,
        thresholds_path=args.mlp_thresholds,
        test_indices_path=args.test_indices,
        batch_size=get_model_batch_size(cfg, "mlp"),
        class_names=class_names
        )
    metrics_rows.append(mlp_metrics_row)
    prediction_rows.extend(mlp_prediction_rows)

    # evaluate cnn
    cnn_metrics_row, cnn_prediction_rows =\
        evaluate_classification_model(
            model_family="cnn",
            config_path=args.config,
            weights_path=args.cnn_weights,
            thresholds_path=args.cnn_thresholds,
            test_indices_path=args.test_indices,
            batch_size=get_model_batch_size(cfg, "cnn"),
            class_names=class_names
            )
    metrics_rows.append(cnn_metrics_row)
    prediction_rows.extend(cnn_prediction_rows)

    # evaluate multiunet
    multiunet_metrics_row, multiunet_seg_row, multiunet_prediction_rows =\
         evaluate_multiunet_model(
            config_path=args.config,
            weights_path=args.multiunet_weights,
            thresholds_path=args.multiunet_thresholds,
            test_indices_path=args.test_indices,
            batch_size=get_model_batch_size(cfg, "multiunet"),
            class_names=class_names
            )
    metrics_rows.append(multiunet_metrics_row)
    segmentation_rows.append(multiunet_seg_row)
    prediction_rows.extend(multiunet_prediction_rows)

    # evaluate attentioncnn when the workflow passes completed artifacts
    if should_evaluate_attentioncnn(args):
        attentioncnn_metrics_row, attentioncnn_seg_row, attentioncnn_prediction_rows =\
             evaluate_attentioncnn_model(
                config_path=args.config,
                classifier_weights_path=args.attentioncnn_weights,
                segmenter_weights_path=args.attentioncnn_segmenter_weights,
                thresholds_path=args.attentioncnn_thresholds,
                test_indices_path=args.test_indices,
                batch_size=get_model_batch_size(cfg, "attentioncnn"),
                class_names=class_names
            )
        metrics_rows.append(attentioncnn_metrics_row)
        segmentation_rows.append(attentioncnn_seg_row)
        prediction_rows.extend(attentioncnn_prediction_rows)
        models_evaluated.append("attentioncnn")

    # convert lists to dfs
    metrics_df = pd.DataFrame(metrics_rows)\
        .sort_values("model_family").reset_index(drop=True)
    predictions_df = pd.DataFrame(prediction_rows)\
        .sort_values(["model_family", "meta_index"]).reset_index(drop=True)
    segmentation_df = pd.DataFrame(segmentation_rows)\
        .sort_values("model_family").reset_index(drop=True)

    # write csvs
    write_dataframe(metrics_df, args.metrics_out)
    write_dataframe(predictions_df, args.predictions_out)
    write_dataframe(segmentation_df, args.segmentation_out)

    summary = {
        "models_evaluated": models_evaluated,
        "class_names": class_names,
        "num_prediction_rows": int(len(predictions_df)),
        "num_metric_rows": int(len(metrics_df)),
        "num_segmentation_rows": int(len(segmentation_df)),
        "metrics_csv": args.metrics_out,
        "predictions_csv": args.predictions_out,
        "segmentation_csv": args.segmentation_out
        }

    write_json(summary, args.summary_out)

    print(f"Saved metrics: {args.metrics_out}")
    print(f"Saved predictions: {args.predictions_out}")
    print(f"Saved segmentation metrics: {args.segmentation_out}")
    print(f"Saved summary: {args.summary_out}")


if __name__ == "__main__":
    main()
