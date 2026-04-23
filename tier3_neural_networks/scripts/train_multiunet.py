#!/usr/bin/env python3

import argparse
import time
from math import ceil

from models.multiunet import build_multiunet
from shared.config_utils import load_config
from shared.eval_utils import (apply_thresholds, mean_dice_score,summarize_classification_metrics)

from shared.json_and_csv_utils import write_json, read_index_file
from shared.keras_utils import (
    compile_multitask_model,
    compute_positive_class_weights,
    get_common_callbacks,
)
from shared.run_model_inference import run_multitask_inference
from shared.tfrecord_utils import build_split_dataset


# note: no hyperparam search at training time, only fitting weights
# classification threshold is fixed at 0.5 at this stage

#CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")        # config file
    parser.add_argument("--train-indices", type=str, required=True)         # train split indices
    parser.add_argument("--val-indices", type=str, required=True)           # val split indices
    parser.add_argument("--weights-out", type=str, required=True)           # output dir for weights
    parser.add_argument("--summary-out", type=str, required=True)           # output dir for summary
    return parser.parse_args()

# build using config's specs
def build_model_from_config(cfg: dict):
    return build_multiunet(
        input_shape=tuple(cfg["data"]["input_shape"]),
        num_classes=int(cfg["data"]["num_classes"]),
        encoder_filters=tuple(cfg["multiunet"]["encoder_filters"]),
        bottleneck_filters=int(cfg["multiunet"]["bottleneck_filters"]),
        decoder_filters=tuple(cfg["multiunet"]["decoder_filters"]),
        cls_hidden_units=tuple(cfg["multiunet"]["cls_hidden_units"]),
        kernel_size=int(cfg["multiunet"]["kernel_size"]),
        dropout_rate=float(cfg["multiunet"]["dropout_rate"]),
        use_batch_norm=bool(cfg["multiunet"]["use_batch_norm"])
        )


def main() -> None:
    # parse args
    args = parse_args()
    
    
    cfg = load_config(args.config)
    batch_size = int(cfg["multiunet"]["batch_size"])
    epochs = int(cfg["training"]["epochs"])
    learning_rate = float(cfg["training"]["learning_rate"])
    patience = int(cfg["training"]["early_stopping_patience"])
    loss_weight_cls = float(cfg["multiunet"]["loss_weight_cls"])
    loss_weight_seg = float(cfg["multiunet"]["loss_weight_seg"])
    class_names = list(cfg["data"]["class_names"])

    # build train and val
    train_indices = read_index_file(args.train_indices)
    val_indices = read_index_file(args.val_indices)

    steps_per_epoch = ceil(len(train_indices) / batch_size)
    validation_steps = ceil(len(val_indices) / batch_size)

    train_ds = build_split_dataset(
        config_path=args.config,
        split_indices_path=args.train_indices,
        batch_size=batch_size,
        shuffle=True,
        task="multitask",
        include_metadata=False
        )

    class_pos_weights = compute_positive_class_weights(
        train_ds,
        num_classes=int(cfg["data"]["num_classes"]),
        target_key="cls"
        )
    train_ds = train_ds.repeat()

    val_ds = build_split_dataset(
        config_path=args.config,
        split_indices_path=args.val_indices,
        batch_size=batch_size,
        shuffle=False,
        task="multitask",
        include_metadata=False
        ).repeat()

    # build model
    model = build_model_from_config(cfg)
    model = compile_multitask_model(
        model,
        learning_rate=learning_rate,
        loss_weight_cls=loss_weight_cls,
        loss_weight_seg=loss_weight_seg,
        class_pos_weights=class_pos_weights
        )

    # train model
    start_time = time.perf_counter()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=get_common_callbacks(weights_out=args.weights_out,patience=patience),
        verbose=1
        )
    elapsed_s = time.perf_counter() - start_time

    # use model on val
    outputs = run_multitask_inference(
        config_path=args.config,
        model_path=args.weights_out,
        split_indices_path=args.val_indices,
        batch_size=batch_size
        )

    y_true_cls = outputs["y_true_cls"]
    y_prob_cls = outputs["y_prob_cls"]
    y_pred_cls = apply_thresholds(y_prob_cls, thresholds=[0.5] * y_prob_cls.shape[1])

    # report metrics
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
        "model_family": "multiunet",
        "train_examples": len(train_indices),
        "val_examples": len(val_indices),
        "epochs_requested": epochs,
        "epochs_ran": len(history.history.get("loss", [])),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "loss_weight_cls": loss_weight_cls,
        "loss_weight_seg": loss_weight_seg,
        "class_pos_weights": class_pos_weights,
        "train_time_s": round(elapsed_s, 2),
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
        "val_classification_metrics_default_threshold_0p5": val_cls_metrics,
        "val_segmentation_metrics": val_seg_metrics
        }

    write_json(summary, args.summary_out)

    print(f"Saved weights: {args.weights_out}")
    print(f"Saved summary: {args.summary_out}")


if __name__ == "__main__":
    main()
