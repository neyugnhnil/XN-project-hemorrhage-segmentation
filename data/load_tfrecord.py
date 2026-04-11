import tensorflow as tf

# one example has...
FEATURE_SPEC = {
    "meta_index": tf.io.FixedLenFeature([], tf.int64),
    "id": tf.io.FixedLenFeature([], tf.string),
    "render_directory": tf.io.FixedLenFeature([], tf.string),
    "seg_label_source": tf.io.FixedLenFeature([], tf.string),
    "x_raw": tf.io.FixedLenFeature([], tf.string),
    "y_cls_raw": tf.io.FixedLenFeature([], tf.string),
    "y_seg_raw": tf.io.FixedLenFeature([], tf.string)
    }
# all scalars right now (serialized)


def parse_example(example_proto: tf.Tensor):

    # make parsed dictionary (still serialized bytes)
    parsed = tf.io.parse_single_example(example_proto, FEATURE_SPEC) 

    # deserialize tensors 
    x = tf.io.parse_tensor(parsed["x_raw"], out_type=tf.float32)
    y_cls = tf.io.parse_tensor(parsed["y_cls_raw"], out_type=tf.float32)
    y_seg = tf.io.parse_tensor(parsed["y_seg_raw"], out_type=tf.float32)

    # ensure shapes
    x = tf.ensure_shape(x, [512, 512, 4])
    y_cls = tf.ensure_shape(y_cls, [6])
    y_seg = tf.ensure_shape(y_seg, [512, 512, 1])

    # metadata dictionary
    metadata = {
        "meta_index": tf.cast(parsed["meta_index"], tf.int32),
        "id": parsed["id"],
        "render_directory": parsed["render_directory"],
        "seg_label_source": parsed["seg_label_source"]
        }

    # target dictionary (y)
    y = {"cls": y_cls, "seg": y_seg}
    return x, y, metadata


def load_tfrecord(tfrecord_path: str, batch_size: int = 4, shuffle: bool = False, include_metadata: bool = False)\
    -> tf.data.Dataset:

    # open file
    ds = tf.data.TFRecordDataset(tfrecord_path)

    # parse every example 
    ds = ds.map(parse_example, num_parallel_calls=tf.data.AUTOTUNE) # -> (x, y, metadata)

    if not include_metadata:
        # if not needed, remove metadata
        ds = ds.map(lambda x, y, metadata: (x, y), num_parallel_calls=tf.data.AUTOTUNE,)

    if shuffle: 
        # shuffle with mixing buffer of size 1024
        ds = ds.shuffle(1024)

    # group into batches
    ds = ds.batch(batch_size)

    # allow prefetch
    ds = ds.prefetch(tf.data.AUTOTUNE)
    
    return ds