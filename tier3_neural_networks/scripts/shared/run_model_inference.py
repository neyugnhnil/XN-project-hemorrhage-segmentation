#!/usr/bin/env python3

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow import keras

from shared.config_utils import load_config
from shared.keras_utils import dice_coefficient, dice_loss
from shared.tfrecord_utils import build_split_dataset

# load saved keras model
def load_keras_model(model_path: str | Path, custom_objects: dict[str, Any] | None = None) -> keras.Model:
    return keras.models.load_model(str(model_path), custom_objects=custom_objects)

# revert metadata tensors to row form
def _metadata_batch_to_rows(metadata_batch: dict[str, tf.Tensor]) -> list[dict[str, Any]]:
    meta_index = metadata_batch["meta_index"].numpy()
    ids = metadata_batch["id"].numpy()
    render_directory = metadata_batch["render_directory"].numpy()
    seg_label_source = metadata_batch["seg_label_source"].numpy()

    rows: list[dict[str, Any]] = []

    for i in range(len(meta_index)):
        rows.append(
            {
                "meta_index": int(meta_index[i]),
                "id": ids[i].decode("utf-8"),
                "render_directory": render_directory[i].decode("utf-8"),
                "seg_label_source": seg_label_source[i].decode("utf-8")
            }
        )

    return rows

# collect outputs/metadata of MLP/CNN inference
def collect_classification_outputs(model: keras.Model, ds: tf.data.Dataset) -> dict[str, Any]:
    y_true_list: list[np.ndarray] = []
    y_prob_list: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []

    for x_batch, y_batch, metadata_batch in ds:
        y_prob_batch = model.predict_on_batch(x_batch)
        y_true_list.append(y_batch.numpy())
        y_prob_list.append(np.asarray(y_prob_batch))
        metadata_rows.extend(_metadata_batch_to_rows(metadata_batch))

    return {
        "y_true_cls": np.concatenate(y_true_list, axis=0),
        "y_prob_cls": np.concatenate(y_prob_list, axis=0),
        "metadata": metadata_rows
        }

# collect outputs/metadata of multiunet inference
def collect_multitask_outputs(model: keras.Model, ds: tf.data.Dataset) -> dict[str, Any]:
    y_true_cls_list: list[np.ndarray] = []
    y_prob_cls_list: list[np.ndarray] = []
    y_true_seg_list: list[np.ndarray] = []
    y_prob_seg_list: list[np.ndarray] = []
    metadata_rows: list[dict[str, Any]] = []

    for x_batch, y_batch, metadata_batch in ds:

        outputs = model.predict_on_batch(x_batch)
    
        y_true_cls_list.append(y_batch["cls"].numpy())
        y_prob_cls_list.append(np.asarray(outputs["cls"]))

        y_true_seg_list.append(y_batch["seg"].numpy())
        y_prob_seg_list.append(np.asarray(outputs["seg"]))

        metadata_rows.extend(_metadata_batch_to_rows(metadata_batch))

    return {
        "y_true_cls": np.concatenate(y_true_cls_list, axis=0),
        "y_prob_cls": np.concatenate(y_prob_cls_list, axis=0),
        "y_true_seg": np.concatenate(y_true_seg_list, axis=0),
        "y_prob_seg": np.concatenate(y_prob_seg_list, axis=0),
        "metadata": metadata_rows
        }

# main inference wrapper for cnn/mlp
def run_classification_inference(
    config_path: str | Path,
    model_path: str | Path,
    split_indices_path: str | Path,
    batch_size: int) -> dict[str, Any]:

    ds = build_split_dataset(
        config_path=config_path,
        split_indices_path=split_indices_path,
        batch_size=batch_size,
        shuffle=False,
        task="classification",
        include_metadata=True
        )

    model = load_keras_model(model_path)
    return collect_classification_outputs(model, ds)

# main inference wrapper for multiunet
def run_multitask_inference(
    config_path: str | Path,
    model_path: str | Path,
    split_indices_path: str | Path,
    batch_size: int) -> dict[str, Any]:

    ds = build_split_dataset(
        config_path=config_path,
        split_indices_path=split_indices_path,
        batch_size=batch_size,
        shuffle=False,
        task="multitask",
        include_metadata=True)

    model = load_keras_model(
        model_path,
        custom_objects={"dice_loss": dice_loss, "dice_coefficient": dice_coefficient}
        )

    return collect_multitask_outputs(model, ds)