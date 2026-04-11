#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers

def build_mlp(
    input_shape=(512, 512, 4),
    num_classes=5, 
    pooled_size=(32, 32),
    hidden_units=(512, 256),
    dropout_rate=0.3,
    use_batch_norm=False
    ):

    # coordinates building of vanilla MLP neural net 
    # x -> downsample -> flatten -> dense -> y
    # x is [512, 512], y is a vector of probs

    inputs = keras.Input(shape=input_shape, name="x")       # symbolic placeholder of input tensors

    # on a 512 x 512 x 4 input tensor, flattening would give 1048576 input features
    # ...so we downsample for mlp by preparing a lower-dimensional tensor to vectorize
    x = layers.Resizing(
        height=pooled_size[0],
        width=pooled_size[1],
        interpolation="bilinear",
        name="resize_down"
        )(inputs)
    x = layers.Flatten(name="flatten")(x)

    # dense hidden layers
    for i, units in enumerate(hidden_units, start=1):
        x = layers.Dense(units, name=f"dense_{i}")(x)                       # applies affine map z=Wx+b
        
        if use_batch_norm:                                          
            x = layers.BatchNormalization(name=f"bn_{i}")(x)                # batch norm step if needed
        x = layers.Activation("relu", name=f"relu_{i}")(x)                  # ReLU(z) = max(0,z)
        
        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x)        # random dropout of activation, maybe

    # final classification output layer, sigmoid activation
    cls = layers.Dense(num_classes, activation="sigmoid", name="cls")(x) 

    # return model object
    return keras.Model(inputs=inputs, outputs=cls, name="mlp")