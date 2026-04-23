#!/usr/bin/env python3

import argparse
import time
from math import ceil

import tensorflow as tf

from models.cnn import build_cnn
from models.unet import build_unet
from shared.config_utils import (
    get_attentioncnn_attention_floor,
    get_attentioncnn_classifier_config,
    get_attentioncnn_segmenter_config,
    load_config,
)
from shared.eval_utils import apply_thresholds, mean_dice_score, summarize_classification_metrics
from shared.json_and_csv_utils import read_index_file, write_json
from shared.keras_utils import (
    compile_classification_model,
    compile_segmentation_model,
    compute_positive_class_weights,
    dice_coefficient,
    dice_loss,
    get_common_callbacks,
)
from shared.run_model_inference import load_keras_model, run_attentioncnn_inference
from shared.tfrecord_utils import build_split_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--train-indices", type=str, required=True)
    parser.add_argument("--val-indices", type=str, required=True)
    parser.add_argument("--weights-out", type=str, required=True)
    parser.add_argument("--segmenter-weights-out", type=str, required=True)
    parser.add_argument("--summary-out", type=str, required=True)
    return parser.parse_args()


def build_segmenter_from_config(cfg: dict):
    segmenter_cfg = get_attentioncnn_segmenter_config(cfg)
    return build_unet(
        input_shape=tuple(cfg["data"]["input_shape"]),
        encoder_filters=tuple(segmenter_cfg["encoder_filters"]),
        bottleneck_filters=int(segmenter_cfg["bottleneck_filters"]),
        decoder_filters=tuple(segmenter_cfg["decoder_filters"]),
        kernel_size=int(segmenter_cfg["kernel_size"]),
        dropout_rate=float(segmenter_cfg["dropout_rate"]),
        use_batch_norm=bool(segmenter_cfg["use_batch_norm"])
        )


def build_classifier_from_config(cfg: dict):
    classifier_cfg = get_attentioncnn_classifier_config(cfg)
    return build_cnn(
        input_shape=tuple(cfg["data"]["input_shape"]),
        num_classes=int(cfg["data"]["num_classes"]),
        conv_filters=tuple(classifier_cfg["conv_filters"]),
        kernel_size=int(classifier_cfg["kernel_size"]),
        dense_units=tuple(classifier_cfg["dense_units"]),
        dropout_rate=float(classifier_cfg["dropout_rate"]),
        use_batch_norm=bool(classifier_cfg["use_batch_norm"]),
        global_pool=str(classifier_cfg["global_pool"])
        )


def apply_segmenter_attention(
    ds: tf.data.Dataset,
    segmenter: tf.keras.Model,
    attention_floor: float) -> tf.data.Dataset:

    segmenter.trainable = False

    def mask_batch(x_batch, y_batch):
        mask_batch = segmenter(x_batch, training=False)
        attention_batch = attention_floor + ((1.0 - attention_floor) * mask_batch)
        return x_batch * tf.cast(attention_batch, x_batch.dtype), y_batch

    return ds.map(mask_batch)


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    train_indices = read_index_file(args.train_indices)
    val_indices = read_index_file(args.val_indices)

    epochs = int(cfg["training"]["epochs"])
    learning_rate = float(cfg["training"]["learning_rate"])
    patience = int(cfg["training"]["early_stopping_patience"])
    class_names = list(cfg["data"]["class_names"])

    segmenter_cfg = get_attentioncnn_segmenter_config(cfg)
    classifier_cfg = get_attentioncnn_classifier_config(cfg)
    attention_floor = get_attentioncnn_attention_floor(cfg)
    segmenter_batch_size = int(segmenter_cfg["batch_size"])
    classifier_batch_size = int(classifier_cfg["batch_size"])

    train_seg_ds = build_split_dataset(
        config_path=args.config,
        split_indices_path=args.train_indices,
        batch_size=segmenter_batch_size,
        shuffle=True,
        task="segmentation",
        include_metadata=False
        ).repeat()

    val_seg_ds = build_split_dataset(
        config_path=args.config,
        split_indices_path=args.val_indices,
        batch_size=segmenter_batch_size,
        shuffle=False,
        task="segmentation",
        include_metadata=False
        ).repeat()

    segmenter = build_segmenter_from_config(cfg)
    segmenter = compile_segmentation_model(segmenter, learning_rate=learning_rate)

    start_time = time.perf_counter()
    segmenter_history = segmenter.fit(
        train_seg_ds,
        validation_data=val_seg_ds,
        epochs=epochs,
        steps_per_epoch=ceil(len(train_indices) / segmenter_batch_size),
        validation_steps=ceil(len(val_indices) / segmenter_batch_size),
        callbacks=get_common_callbacks(
            weights_out=args.segmenter_weights_out,
            patience=patience
            ),
        verbose=1
        )
    segmenter_elapsed_s = time.perf_counter() - start_time
    segmenter = load_keras_model(
        args.segmenter_weights_out,
        custom_objects={"dice_loss": dice_loss, "dice_coefficient": dice_coefficient}
        )

    train_cls_ds = build_split_dataset(
        config_path=args.config,
        split_indices_path=args.train_indices,
        batch_size=classifier_batch_size,
        shuffle=True,
        task="classification",
        include_metadata=False
        )

    class_pos_weights = compute_positive_class_weights(
        train_cls_ds,
        num_classes=int(cfg["data"]["num_classes"])
        )

    train_cls_ds = apply_segmenter_attention(
        train_cls_ds,
        segmenter,
        attention_floor=attention_floor
        ).repeat()

    val_cls_ds = build_split_dataset(
        config_path=args.config,
        split_indices_path=args.val_indices,
        batch_size=classifier_batch_size,
        shuffle=False,
        task="classification",
        include_metadata=False
        )
    val_cls_ds = apply_segmenter_attention(
        val_cls_ds,
        segmenter,
        attention_floor=attention_floor
        ).repeat()

    classifier = build_classifier_from_config(cfg)
    classifier = compile_classification_model(
        classifier,
        learning_rate=learning_rate,
        class_pos_weights=class_pos_weights
        )

    start_time = time.perf_counter()
    classifier_history = classifier.fit(
        train_cls_ds,
        validation_data=val_cls_ds,
        epochs=epochs,
        steps_per_epoch=ceil(len(train_indices) / classifier_batch_size),
        validation_steps=ceil(len(val_indices) / classifier_batch_size),
        callbacks=get_common_callbacks(weights_out=args.weights_out, patience=patience),
        verbose=1
        )
    classifier_elapsed_s = time.perf_counter() - start_time

    outputs = run_attentioncnn_inference(
        config_path=args.config,
        classifier_model_path=args.weights_out,
        segmenter_model_path=args.segmenter_weights_out,
        split_indices_path=args.val_indices,
        batch_size=classifier_batch_size
        )

    y_true_cls = outputs["y_true_cls"]
    y_prob_cls = outputs["y_prob_cls"]
    y_pred_cls = apply_thresholds(y_prob_cls, thresholds=[0.5] * y_prob_cls.shape[1])

    val_cls_metrics = summarize_classification_metrics(
        y_true=y_true_cls,
        y_prob=y_prob_cls,
        y_pred=y_pred_cls,
        class_names=class_names
        )

    val_seg_metrics = {
        "mean_dice_threshold_0p5": mean_dice_score(
            outputs["y_true_seg"],
            outputs["y_prob_seg"],
            threshold=0.5,
            eps=1.0e-7
            )
        }

    summary = {
        "model_family": "attentioncnn",
        "train_examples": len(train_indices),
        "val_examples": len(val_indices),
        "epochs_requested": epochs,
        "segmenter_epochs_ran": len(segmenter_history.history.get("loss", [])),
        "classifier_epochs_ran": len(classifier_history.history.get("loss", [])),
        "learning_rate": learning_rate,
        "segmenter_batch_size": segmenter_batch_size,
        "classifier_batch_size": classifier_batch_size,
        "segmenter_config": segmenter_cfg,
        "classifier_config": classifier_cfg,
        "attention_floor": attention_floor,
        "attention_operation": "x * (attention_floor + (1 - attention_floor) * predicted_segmentation_probability)",
        "class_pos_weights": class_pos_weights,
        "segmenter_train_time_s": round(segmenter_elapsed_s, 2),
        "classifier_train_time_s": round(classifier_elapsed_s, 2),
        "segmenter_history": {
            k: [float(vv) for vv in v]
            for k, v in segmenter_history.history.items()
            },
        "classifier_history": {
            k: [float(vv) for vv in v]
            for k, v in classifier_history.history.items()
            },
        "val_metrics_default_threshold_0p5": val_cls_metrics,
        "val_segmentation_metrics": val_seg_metrics
        }

    write_json(summary, args.summary_out)

    print(f"Saved segmenter weights: {args.segmenter_weights_out}")
    print(f"Saved classifier weights: {args.weights_out}")
    print(f"Saved summary: {args.summary_out}")


if __name__ == "__main__":
    main()
