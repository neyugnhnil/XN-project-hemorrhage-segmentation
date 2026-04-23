#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers

from models.multiunet import _decoder, _encoder, _unet_convs


def build_unet(
    input_shape=(512, 512, 4),
    encoder_filters=(32, 64, 128, 256),
    bottleneck_filters=512,
    decoder_filters=(256, 128, 64, 32),
    kernel_size=3,
    dropout_rate=0.3,
    use_batch_norm=True
    ):

    if len(encoder_filters) != len(decoder_filters):
        raise ValueError("encoder_filters and decoder_filters must have the same length")

    inputs = keras.Input(shape=input_shape, name="x")

    skips, x = _encoder(inputs, encoder_filters, kernel_size, use_batch_norm)
    x = _unet_convs(x, bottleneck_filters, kernel_size, use_batch_norm, name="bottleneck")

    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name="bottleneck_dropout")(x)

    x = _decoder(x, skips, decoder_filters, kernel_size, use_batch_norm)
    seg = layers.Conv2D(1, 1, activation="sigmoid", name="seg")(x)

    return keras.Model(inputs=inputs, outputs=seg, name="unet")
