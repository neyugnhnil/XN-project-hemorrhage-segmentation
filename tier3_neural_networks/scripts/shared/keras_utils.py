#!/usr/bin/env python3

from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from shared.json_and_csv_utils import ensure_parent_dir

# optimizer
def make_optimizer(learning_rate: float) -> keras.optimizers.Optimizer:
    # defaulting to Adam
    return keras.optimizers.Adam(learning_rate=learning_rate)


# callbacks
def make_early_stopping(patience: int) -> keras.callbacks.EarlyStopping:
    return keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
        )

def make_model_checkpoint(weights_out: str | Path) -> keras.callbacks.ModelCheckpoint:
    ensure_parent_dir(weights_out)
    return keras.callbacks.ModelCheckpoint(
        filepath=str(weights_out),
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False,
        verbose=1
        )

def make_csv_logger(log_path: str | Path) -> keras.callbacks.CSVLogger:
    ensure_parent_dir(log_path)
    return keras.callbacks.CSVLogger(str(log_path))

# bundle callbacks together
def get_common_callbacks(
    weights_out: str | Path,
    patience: int,
    csv_log_path: str | Path | None = None
    ) -> list[keras.callbacks.Callback]:

    callbacks: list[keras.callbacks.Callback] = [
        make_early_stopping(patience),
        make_model_checkpoint(weights_out)
        ]

    if csv_log_path is not None:
        callbacks.append(make_csv_logger(csv_log_path))

    return callbacks

# dice loss function for multiunet compilation
def dice_coefficient(y_true: tf.Tensor, y_pred: tf.Tensor, eps: float = 1.0e-7) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true_f = tf.reshape(y_true, [tf.shape(y_true)[0], -1])
    y_pred_f = tf.reshape(y_pred, [tf.shape(y_pred)[0], -1])

    intersection = tf.reduce_sum(y_true_f * y_pred_f, axis=1)
    denom = tf.reduce_sum(y_true_f, axis=1) + tf.reduce_sum(y_pred_f, axis=1)

    dice = (2.0 * intersection + eps) / (denom + eps)
    return tf.reduce_mean(dice)

def dice_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    return 1.0 - dice_coefficient(y_true, y_pred)


def dice_bce_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    return dice_loss(y_true, y_pred) + tf.reduce_mean(bce)


def compute_positive_class_weights(
    ds: tf.data.Dataset,
    num_classes: int,
    target_key: str | None = None
    ) -> list[float]:
    positive_counts = tf.zeros([num_classes], dtype=tf.float32)
    total_count = tf.constant(0.0, dtype=tf.float32)

    for _, y_batch in ds:
        y_cls = y_batch[target_key] if target_key is not None else y_batch
        y_cls = tf.cast(y_cls, tf.float32)
        positive_counts += tf.reduce_sum(y_cls, axis=0)
        total_count += tf.cast(tf.shape(y_cls)[0], tf.float32)

    negative_counts = total_count - positive_counts
    positive_counts = tf.maximum(positive_counts, 1.0)
    return [float(weight) for weight in (negative_counts / positive_counts).numpy()]


@keras.utils.register_keras_serializable(package="tier3")
class WeightedBinaryCrossentropy(keras.losses.Loss):
    def __init__(self, pos_weights: list[float], name: str = "weighted_binary_crossentropy"):
        super().__init__(name=name)
        self.pos_weights = [float(weight) for weight in pos_weights]
        self.pos_weights_tensor = tf.constant(self.pos_weights, dtype=tf.float32)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1.0e-7, 1.0 - 1.0e-7)

        loss_pos = -y_true * tf.math.log(y_pred) * self.pos_weights_tensor
        loss_neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(loss_pos + loss_neg)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({"pos_weights": self.pos_weights})
        return config

# compilation pathways
def compile_classification_model(
    model: keras.Model, 
    learning_rate: float,
    class_pos_weights: list[float] | None = None) -> keras.Model:

    loss = (
        WeightedBinaryCrossentropy(class_pos_weights)
        if class_pos_weights is not None
        else "binary_crossentropy"
        )

    model.compile(
        optimizer=make_optimizer(learning_rate),
        loss=loss
        )

    return model


def compile_multitask_model(
    model: keras.Model, 
    learning_rate: float, 
    loss_weight_cls: float, 
    loss_weight_seg: float,
    class_pos_weights: list[float] | None = None) -> keras.Model:

    cls_loss = (
        WeightedBinaryCrossentropy(class_pos_weights)
        if class_pos_weights is not None
        else "binary_crossentropy"
        )

    model.compile(
        optimizer=make_optimizer(learning_rate),
        loss={"cls": cls_loss, "seg": dice_bce_loss},
        loss_weights={"cls": loss_weight_cls, "seg": loss_weight_seg}
        )

    return model


def compile_segmentation_model(
    model: keras.Model,
    learning_rate: float) -> keras.Model:

    model.compile(
        optimizer=make_optimizer(learning_rate),
        loss=dice_loss
        )

    return model
