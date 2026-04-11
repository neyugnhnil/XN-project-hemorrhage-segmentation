#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers


def _cnn_convs(x, filters, kernel_size, use_batch_norm, name):
    # helper that builds a convolution block with two convolutions

    # first convolution
    x = layers.Conv2D(filters, kernel_size, padding="same", name=f"{name}_conv1")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name}_bn1")(x)            # batch norm step if needed
    x = layers.Activation("relu", name=f"{name}_relu1")(x)

    # second convolution
    x = layers.Conv2D(filters, kernel_size, padding="same", name=f"{name}_conv2")(x)
    if use_batch_norm:
        x = layers.BatchNormalization(name=f"{name}_bn2")(x)            # batch norm step if needed
    x = layers.Activation("relu", name=f"{name}_relu2")(x)

    # max pooling
    x = layers.MaxPooling2D(pool_size=2, name=f"{name}_pool")(x)

    # return transformed feature tensor
    return x


def build_cnn(
    input_shape=(512, 512, 4),
    num_classes=5,
    conv_filters=(32, 64, 128, 256),
    kernel_size=3,
    dense_units=(256,),
    dropout_rate=0.3,
    use_batch_norm=True,
    global_pool="avg"
    ):

    # x -> conv-pool blocks -> global pooling -> dense head -> y
    # x is [512, 512], y is a vector of probs

    inputs = keras.Input(shape=input_shape, name="x")               # symbolic placeholder of input tensors
    x = inputs

    # convolutions
    for i, filters in enumerate(conv_filters, start=1):
        # will loop over conv_filters list provided 
        # each later block should get more spatially general 
        x = _cnn_convs(x, filters, kernel_size, use_batch_norm, name=f"block_{i}")

    # global pooling (producing one scalar per channel from last convolutional block)
    if global_pool == "avg":
        x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    else:
        x = layers.GlobalMaxPooling2D(name="global_max_pool")(x)

    # dense layers
    for i, units in enumerate(dense_units, start=1):
        x = layers.Dense(units, name=f"dense_{i}")(x)                               # applies affine map z=Wx+b
        if use_batch_norm:
            x = layers.BatchNormalization(name=f"dense_bn_{i}")(x)                  # batch norm step if needed
        x = layers.Activation("relu", name=f"dense_relu_{i}")(x)                    # ReLU(z) = max(0,z)
        
        if dropout_rate > 0:                                                        # random dropout of activation, maybe
            x = layers.Dropout(dropout_rate, name=f"dense_dropout_{i}")(x)

    # final classification output layer, sigmoid activation
    cls = layers.Dense(num_classes, activation="sigmoid", name="cls")(x)
    
    # return model object
    return keras.Model(inputs=inputs, outputs=cls, name="cnn")