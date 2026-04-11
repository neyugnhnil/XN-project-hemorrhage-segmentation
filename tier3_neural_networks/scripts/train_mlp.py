#!/usr/bin/env python3

import argparse
import time

from math import ceil

from models.mlp import build_mlp
from shared.config_utils import load_config
from shared.eval_utils import apply_thresholds, summarize_classification_metrics
from shared.json_and_csv_utils import write_json, read_index_file
from shared.keras_utils import compile_classification_model, get_common_callbacks
from shared.run_model_inference import run_classification_inference
from shared.tfrecord_utils import build_split_dataset


# note: no hyperparam search at training time at the moment, only fitting weights
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
    return build_mlp(
        input_shape=tuple(cfg["data"]["input_shape"]),
        num_classes=int(cfg["data"]["num_classes"]),
        pooled_size=tuple(cfg["mlp"]["pooled_size"]),
        hidden_units=tuple(cfg["mlp"]["hidden_units"]),
        dropout_rate=float(cfg["mlp"]["dropout_rate"]),
        use_batch_norm=bool(cfg["mlp"]["use_batch_norm"])
        )


def main() -> None:
    # parse args
    args = parse_args()

    # load config specs
    cfg = load_config(args.config)
    batch_size = int(cfg["mlp"]["batch_size"])
    epochs = int(cfg["training"]["epochs"])
    learning_rate = float(cfg["training"]["learning_rate"])
    patience = int(cfg["training"]["early_stopping_patience"])
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
        task="classification",
        include_metadata=False
        ).repeat()

    val_ds = build_split_dataset(
        config_path=args.config,
        split_indices_path=args.val_indices,
        batch_size=batch_size,
        shuffle=False,
        task="classification",
        include_metadata=False
        ).repeat()

    # build model
    model = build_model_from_config(cfg)
    model = compile_classification_model(model, learning_rate=learning_rate)

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
    outputs = run_classification_inference(
        config_path=args.config,
        model_path=args.weights_out,
        split_indices_path=args.val_indices,
        batch_size=batch_size
        )
    
    y_true = outputs["y_true_cls"]
    y_prob = outputs["y_prob_cls"]
    y_pred = apply_thresholds(y_prob, thresholds=[0.5] * y_prob.shape[1])

    # report metrics
    val_metrics = summarize_classification_metrics(
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
        class_names=class_names
        )

    summary = {
        "model_family": "mlp",
        "train_examples": len(train_indices),
        "val_examples": len(val_indices),
        "epochs_requested": epochs,
        "epochs_ran": len(history.history.get("loss", [])),
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "train_time_s": round(elapsed_s, 2),
        "history": {k: [float(vv) for vv in v] for k, v in history.history.items()},
        "val_metrics_default_threshold_0p5": val_metrics
        }

    write_json(summary, args.summary_out)

    print(f"Saved weights: {args.weights_out}")
    print(f"Saved summary: {args.summary_out}")


if __name__ == "__main__":
    main()