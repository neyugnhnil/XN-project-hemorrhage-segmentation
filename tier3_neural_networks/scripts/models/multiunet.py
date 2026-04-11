#!/usr/bin/env python3

from tensorflow import keras
from tensorflow.keras import layers

# summary:
# x -> encoder -> feature map F
# F -> decoder -> (soft) segmentation map M
# F, M -> classification head -> y

def _unet_convs(x, filters, kernel_size, use_batch_norm, name):

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

    # return transformed ftr tensor
    return x


def _encoder(x, encoder_filters, kernel_size, use_batch_norm):
    # x is one feature tensor

    # this list will store intermediate ftr maps used later by the decoder as skip connections
    skips = [] 
    # before each downsampling step, the encoder saves a feature map at that resolution
    # the decoder uses these saved maps to recover fine spatial detail loss during the compression below

    for i, filters in enumerate(encoder_filters, start=1):
        # loops through filters in order and learns ftr maps, then pools
        x = _unet_convs(x, filters, kernel_size, use_batch_norm, name=f"enc_{i}")  
        skips.append(x)
        x = layers.MaxPooling2D(pool_size=2, name=f"enc_{i}_pool")(x)

    # return ftr maps at each resolution and final compressed ftr tensor
    return skips, x


def _decoder(x, skips, decoder_filters, kernel_size, use_batch_norm):
    # x is one feature tensor

    for i, (skip, filters) in enumerate(zip(reversed(skips), decoder_filters), start=1):
        
        # upsample
        x = layers.UpSampling2D(size=2, interpolation="bilinear", name=f"dec_{i}_up")(x)

        # concatenate with skip tensor 
        x = layers.Concatenate(name=f"dec_{i}_concat")([x, skip])

        # another two convolutions to blend into new feature map
        x = _unet_convs(x, filters, kernel_size, use_batch_norm, name=f"dec_{i}")
    
    # return final tensor (soft map)
    return x


def build_multiunet(
    input_shape=(512, 512, 4),
    num_classes=5,
    encoder_filters=(32, 64, 128, 256),
    bottleneck_filters=512,
    decoder_filters=(256, 128, 64, 32),
    kernel_size = 3,
    cls_hidden_units=(512, 256),
    dropout_rate=0.3,
    use_batch_norm=True
    ):

    if len(encoder_filters) != len(decoder_filters):
        raise ValueError("encoder_filters and decoder_filters must have the same length")

    inputs = keras.Input(shape=input_shape, name="x")                           # symbolic placeholder of input tensors

    # encoder
    skips, x = _encoder(inputs, encoder_filters, kernel_size, use_batch_norm)                

    # bottleneck
    # apply one more convolution block at most compressed spatial scale
    x = _unet_convs(x, bottleneck_filters, kernel_size, use_batch_norm, name="bottleneck")
    if dropout_rate > 0:                                                       # optional dropout
        x = layers.Dropout(dropout_rate, name="bottleneck_dropout")(x)
    
    # save a late feature map (F)
    late_features = x 

    # decoder
    x = _decoder(x, skips, decoder_filters, kernel_size, use_batch_norm)

    # produce segmentation output (M)
    seg = layers.Conv2D(1, 1, activation="sigmoid", name="seg")(x)

    # downsample M to F's dimensions
    feat_h = late_features.shape[1]
    feat_w = late_features.shape[2]
    seg_down = layers.Resizing(
        height=feat_h,
        width=feat_w,
        interpolation="bilinear",
        name="seg_down_for_cls"
        )(seg)

    # global pooled feature summary vector
    global_feature_pool = layers.GlobalAveragePooling2D(name="global_feature_pool")(late_features)

    # global pooled feature summary vector weighted by segmentation probability mask
    attended_features = layers.Multiply(name="attended_features")([late_features, seg_down])
    attention_feature_pool = layers.GlobalAveragePooling2D(name="attention_feature_pool")(attended_features)

    # concatenate the two summary vectors to make x_cls
    x_cls = layers.Concatenate(name="cls_feature_concat")([global_feature_pool, attention_feature_pool])

    # classification dense head using feature vector
    for i, units in enumerate(cls_hidden_units, start=1):
        x_cls = layers.Dense(units, name=f"dense_{i}")(x_cls)                       # applies affine map z=Wx+b
        
        if use_batch_norm:                                          
            x_cls = layers.BatchNormalization(name=f"bn_{i}")(x_cls)                # batch norm step if needed
        x_cls = layers.Activation("relu", name=f"relu_{i}")(x_cls)                  # ReLU(z) = max(0,z)
        
        if dropout_rate > 0:
            x_cls = layers.Dropout(dropout_rate, name=f"dropout_{i}")(x_cls)        # random dropout of activation, maybe
    # similar in spirit to our MLP vanilla model, but using features learned from segmentation

    # final output layer
    cls = layers.Dense(num_classes, activation="sigmoid", name="cls")(x_cls)        

    # return multitask model
    return keras.Model(inputs=inputs, outputs={"cls": cls, "seg": seg}, name="multiunet")