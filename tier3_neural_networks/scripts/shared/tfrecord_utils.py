#!/usr/bin/env python3

import tensorflow as tf
from shared.config_utils import get_resolved_paths, load_config
from shared.json_and_csv_utils import read_index_file

FEATURE_SPEC = {
    "meta_index": tf.io.FixedLenFeature([], tf.int64),
    "id": tf.io.FixedLenFeature([], tf.string),
    "render_directory": tf.io.FixedLenFeature([], tf.string),
    "seg_label_source": tf.io.FixedLenFeature([], tf.string),
    "x_raw": tf.io.FixedLenFeature([], tf.string),
    "y_cls_raw": tf.io.FixedLenFeature([], tf.string),
    "y_seg_raw": tf.io.FixedLenFeature([], tf.string)
    }

# the main export is build_split_dataset()

def parse_example(
    example_proto: tf.Tensor,
    input_shape: tuple[int, int, int] = (512, 512, 4),
    seg_shape: tuple[int, int, int] = (512, 512, 1),
    drop_any: bool = True
    ) -> tuple[tf.Tensor, dict[str, tf.Tensor], dict[str, tf.Tensor]]:
    # (same function as in load_tfrecord.py)

    # make parsed dictionary (still serialized bytes)
    parsed = tf.io.parse_single_example(example_proto, FEATURE_SPEC)

    # deserialize tensors
    x = tf.io.parse_tensor(parsed["x_raw"], out_type=tf.float32)
    y_cls = tf.io.parse_tensor(parsed["y_cls_raw"], out_type=tf.float32)
    y_seg = tf.io.parse_tensor(parsed["y_seg_raw"], out_type=tf.float32)

    # ensure shapes before dropping any
    x = tf.ensure_shape(x, input_shape)
    y_cls = tf.ensure_shape(y_cls, [6])
    y_seg = tf.ensure_shape(y_seg, seg_shape)

    # drop "any" from classification labels
    if drop_any:
        y_cls = y_cls[1:]

    # ensure final classification shape
    y_cls = tf.ensure_shape(y_cls, [5])

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


def filter_by_meta_indices(ds: tf.data.Dataset, meta_indices: list[int]) -> tf.data.Dataset:
    
    index_set = tf.constant(sorted(set(meta_indices)), dtype=tf.int32)

    def keep_example(x, y, metadata):
        meta_index = metadata["meta_index"]
        return tf.reduce_any(tf.equal(index_set, meta_index))
    
    return ds.filter(keep_example)


def make_classification_only(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(
        lambda x, y, metadata: (x, y["cls"], metadata), num_parallel_calls=tf.data.AUTOTUNE)


def make_multitask(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(
        lambda x, y, metadata: (x, {"cls": y["cls"], "seg": y["seg"]}, metadata),
        num_parallel_calls=tf.data.AUTOTUNE
        )


def finalize_dataset(ds: tf.data.Dataset, batch_size: int, shuffle: bool, include_metadata: bool)\
     -> tf.data.Dataset:

    if not include_metadata:
        ds = ds.map(
            lambda x, y, metadata: (x, y),
            num_parallel_calls=tf.data.AUTOTUNE
            )

    if shuffle:
        ds = ds.shuffle(1024)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_split_dataset(
    config_path: str, 
    split_indices_path: str, 
    batch_size: int,
    shuffle: bool, 
    task: str, 
    include_metadata: bool = False
    ) -> tf.data.Dataset:

    cfg = load_config(config_path)
    resolved_paths = get_resolved_paths(cfg, config_path)

    input_shape = tuple(cfg["data"]["input_shape"])
    seg_shape = tuple(cfg["data"]["seg_shape"])
    split_indices = read_index_file(split_indices_path)

    # load unbatched dataset with metadata first
    ds = tf.data.TFRecordDataset(str(resolved_paths["tfrecord"]))
    ds = ds.map(
        lambda ex: parse_example(ex,input_shape,seg_shape,drop_any=True),
        num_parallel_calls=tf.data.AUTOTUNE
        )

    # keep only examples from this split
    ds = filter_by_meta_indices(ds, split_indices)

    # choose target format for the model family
    if task == "classification":
        ds = make_classification_only(ds)
    elif task == "multitask":
        ds = make_multitask(ds)
    else:
        raise ValueError("task must be 'classification' or 'multitask'")

    # final batching / metadata stripping
    ds = finalize_dataset(
        ds=ds,
        batch_size=batch_size,
        shuffle=shuffle,
        include_metadata=include_metadata
        )

    return ds