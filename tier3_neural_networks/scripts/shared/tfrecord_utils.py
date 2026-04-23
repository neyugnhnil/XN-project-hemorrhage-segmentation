#!/usr/bin/env python3

import struct

import tensorflow as tf
from shared.config_utils import get_resolved_paths, load_config
from shared.json_and_csv_utils import read_index_file

META_FEATURE_SPEC = {
    "meta_index": tf.io.FixedLenFeature([], tf.int64)
    }

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

def get_example_meta_index(example_proto: bytes) -> int:
    parsed = tf.io.parse_single_example(example_proto, META_FEATURE_SPEC)
    return int(parsed["meta_index"].numpy())


def build_record_location_index(tfrecord_path) -> dict[int, tuple[int, int]]:
    record_locations = {}

    with open(tfrecord_path, "rb") as f:
        while True:
            record_offset = f.tell()
            length_bytes = f.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise ValueError(f"Truncated TFRecord length header in {tfrecord_path}")

            record_length = struct.unpack("<Q", length_bytes)[0]
            length_crc = f.read(4)
            record = f.read(record_length)
            data_crc = f.read(4)

            if len(length_crc) != 4 or len(record) != record_length or len(data_crc) != 4:
                raise ValueError(f"Truncated TFRecord record in {tfrecord_path}")

            meta_index = get_example_meta_index(record)
            if meta_index in record_locations:
                raise ValueError(f"Duplicate meta_index in TFRecord: {meta_index}")
            record_locations[meta_index] = (record_offset + 12, record_length)

    return record_locations


def make_raw_split_dataset(
    tfrecord_path,
    meta_indices: list[int],
    shuffle: bool,
    seed: int
    ) -> tf.data.Dataset:

    record_locations = build_record_location_index(tfrecord_path)
    missing_indices = sorted(set(meta_indices) - set(record_locations))
    if missing_indices:
        raise ValueError(
            "Split contains meta_index values not present in TFRecord, "
            f"e.g. {missing_indices[:10]}"
            )

    ds = tf.data.Dataset.from_tensor_slices(tf.constant(meta_indices, dtype=tf.int64))

    if shuffle:
        # This shuffles tiny integer ids instead of decoded image/mask tensors.
        ds = ds.shuffle(
            buffer_size=len(meta_indices),
            seed=seed,
            reshuffle_each_iteration=True
            )

    tfrecord_path = str(tfrecord_path)

    def read_record(meta_index):
        meta_index = int(meta_index.numpy())
        record_offset, record_length = record_locations[meta_index]
        with open(tfrecord_path, "rb") as f:
            f.seek(record_offset)
            return f.read(record_length)

    def read_record_tensor(meta_index):
        record = tf.py_function(read_record, [meta_index], Tout=tf.string)
        record.set_shape([])
        return record

    return ds.map(read_record_tensor, num_parallel_calls=tf.data.AUTOTUNE)

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

    # ensure shapes before dropping "any"
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


def make_segmentation_only(ds: tf.data.Dataset) -> tf.data.Dataset:
    return ds.map(
        lambda x, y, metadata: (x, y["seg"], metadata),
        num_parallel_calls=tf.data.AUTOTUNE
        )


def finalize_dataset(
    ds: tf.data.Dataset,
    batch_size: int,
    include_metadata: bool)\
     -> tf.data.Dataset:

    if not include_metadata:
        ds = ds.map(
            lambda x, y, metadata: (x, y),
            num_parallel_calls=tf.data.AUTOTUNE
            )

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

    # load raw examples by split index order. 
    # for training, shuffle the integer meta_index stream before reading/parsing
    ds = make_raw_split_dataset(
        tfrecord_path=resolved_paths["tfrecord"],
        meta_indices=split_indices,
        shuffle=shuffle,
        seed=int(cfg["seed"])
        )

    ds = ds.map(
        lambda ex: parse_example(ex,input_shape,seg_shape,drop_any=True),
        num_parallel_calls=tf.data.AUTOTUNE
        )

    # choose target format for the model family
    if task == "classification":
        ds = make_classification_only(ds)
    elif task == "multitask":
        ds = make_multitask(ds)
    elif task == "segmentation":
        ds = make_segmentation_only(ds)
    else:
        raise ValueError("task must be 'classification', 'multitask', or 'segmentation'")

    # final batching / metadata stripping
    ds = finalize_dataset(
        ds=ds,
        batch_size=batch_size,
        include_metadata=include_metadata
        )

    return ds
