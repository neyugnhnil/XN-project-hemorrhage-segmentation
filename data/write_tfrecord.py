#!/usr/bin/env python3
"""
builds a TFRecord dataset from subset_metafile.csv.

each TFRecord example has:
- meta_index: int
- id: string
- render_directory: string
- x: float32 tensor of shape [H, W, 4]
    the 4 channels are:
        0 = brain_bone_window
        1 = brain_window
        2 = max_contrast_window
        3 = subdural_window
- y_cls: float32 tensor of shape [6]
    labels are: 
        [any, epidural, intraparenchymal, intraventricular, subarachnoid, subdural]
- y_seg: float32 tensor of shape [H, W, 1]
        binary mask rasterized from seg_polygon
- seg_label_source: string

outputs:
- data/cases.tfrecord
- data/cases_manifest.csv

notes:
- assumes seg_polygon is a JSON string like:
    [{"x": 0.39, "y": 0.73}, {"x": 0.40, "y": 0.72}, ...]
- assumes the polygon is aligned with the rendered JPG geometry
"""


import json
from pathlib import Path

import tensorflow as tf
import cv2
import numpy as np
import pandas as pd
import ast


## CONSTANTS ##
DATA_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DATA_ROOT.parent
METAFILE_CSV = DATA_ROOT / "subset_metafile.csv"
OUT_TFRECORD = DATA_ROOT / "cases.tfrecord"
OUT_MANIFEST = DATA_ROOT / "cases_manifest.csv"

IMAGE_SIZE = (512, 512)
CLASS_LABEL_COLUMNS = ["any","epidural","intraparenchymal","intraventricular","subarachnoid","subdural"]
WINDOW_PATH_COLUMNS = ["brain_bone_window_path","brain_window_path","max_contrast_window_path","subdural_window_path"]
REQUIRED_COLUMNS = ["id","render_directory","seg_label_source","seg_polygon",*CLASS_LABEL_COLUMNS,*WINDOW_PATH_COLUMNS,]


## HELPERS ##

def load_grayscale_image(path: str, image_size: tuple[int, int]) -> np.ndarray:
    # reads one JPG as grayscale, resizes, and scales to [0, 1].
    # returns shape [H, W], dtype float32.

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, image_size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return img

def _bytes_feature(value: bytes) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value: int) -> tf.train.Feature:
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _serialize_tensor(array: np.ndarray) -> bytes:
    return tf.io.serialize_tensor(tf.convert_to_tensor(array)).numpy()

## POLYGON TO MASK ##

def parse_polygon_points(seg_polygon):
    # takes polygon string and tries its best to make a python list of points

    s = str(seg_polygon).strip() 

    # idea case: normal JSON 
    try:
        data = json.loads(s) # success
    except json.JSONDecodeError:
        # failed. next method.
        data = None

    # if that failed, try other things

    # try removing one layer of outer quotes and fixing doubled quotes
    if data is None:
        cleaned = s
        if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
            cleaned = cleaned[1:-1]
        cleaned = cleaned.replace('""', '"')
        try:
            data = json.loads(cleaned) # success
        except json.JSONDecodeError:
            # failed. next method.
            data = None

    # final fallback: check if it's some sort of python literal
    if data is None:
        try:
            data = ast.literal_eval(s) # success
            #print("ast useful")
        except Exception:
            cleaned = s
            if len(cleaned) >= 2 and cleaned[0] == '"' and cleaned[-1] == '"':
                cleaned = cleaned[1:-1]
            cleaned = cleaned.replace('""', '"')
            
            data = ast.literal_eval(cleaned) # success
            #print("ast useful")

    if not isinstance(data, list):
        return [] # give up

    return data

def polygon_string_to_mask(seg_polygon: str, image_size: tuple[int, int]) -> np.ndarray:
    # convert a polygon string into a binary mask of shape [H, W], dtype float32.
    # assumes the polygon points are normalized to between 0 and 1

    width, height = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    points = parse_polygon_points(seg_polygon)
    if not points:
        return mask.astype(np.float32)

    pts = []
    for p in points:
        if not isinstance(p, dict):
            continue
        if "x" not in p or "y" not in p:
            continue
        
        # converts normalized positions to pixel coordinates
        x = float(p["x"]) * (width - 1)
        y = float(p["y"]) * (height - 1)

        # avoid spilling
        x = np.clip(x, 0, width - 1)
        y = np.clip(y, 0, height - 1)

        pts.append([x, y])

    if len(pts) < 3:
        return mask.astype(np.float32)

    pts = np.array(pts, dtype=np.int32)

    # fill the polygon
    cv2.fillPoly(mask, [pts], 1)

    # return float array
    return mask.astype(np.float32)

## CASE BUILDING ##

def build_case_arrays(row, image_size: tuple[int, int]):
    # build one case from one metafile row.

    channels = []

    # create raster map
    for col in WINDOW_PATH_COLUMNS:
        img = load_grayscale_image(PROJECT_ROOT / getattr(row, col), image_size) # [H, W]
        channels.append(img)

    # stack them into x
    x = np.stack(channels, axis=-1).astype(np.float32)  # [H, W, 4]

    # class labels (vector like [1., 0., 1., 0., 0., 1.])
    y_cls = np.array(
        [getattr(row, col) for col in CLASS_LABEL_COLUMNS],
        dtype=np.float32
        ) # [6]

    # segmentation labels (binary mask)
    y_seg = polygon_string_to_mask(row.seg_polygon, image_size)  # [H, W]
    y_seg = np.expand_dims(y_seg, axis=-1).astype(np.float32)       # [H, W, 1]
    # one channel

    return x, y_cls, y_seg

def make_tf_example(row, image_size: tuple[int, int]) -> tf.train.Example:
    # convert one metafile row (as a tuple) into a tf.train.Example
    
    # get x, y_cls, and y_seg (serialized tensors)
    x, y_cls, y_seg = build_case_arrays(row, image_size) 

    # make feature dictionary
    feature = {
        "meta_index": _int64_feature(int(row.meta_index)),
        "id": _bytes_feature(str(row.id).encode("utf-8")),
        "render_directory": _bytes_feature(str(row.render_directory).encode("utf-8")),
        "seg_label_source": _bytes_feature(str(row.seg_label_source).encode("utf-8")),
        "height": _int64_feature(int(x.shape[0])),
        "width": _int64_feature(int(x.shape[1])),
        "num_channels": _int64_feature(int(x.shape[2])),
        "num_class_labels": _int64_feature(int(y_cls.shape[0])),
        "x_raw": _bytes_feature(_serialize_tensor(x)),
        "y_cls_raw": _bytes_feature(_serialize_tensor(y_cls)),
        "y_seg_raw": _bytes_feature(_serialize_tensor(y_seg)),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))

## MAIN ##

def main() -> None:
    meta_df = pd.read_csv(METAFILE_CSV)
    
    # make stable index row
    meta_df = meta_df.reset_index(names="meta_index") 
   
    manifest_rows = []
    with tf.io.TFRecordWriter(str(OUT_TFRECORD)) as writer:
        for row in meta_df.itertuples(index=False):

            # serialize example
            example = make_tf_example(row, IMAGE_SIZE)
            writer.write(example.SerializeToString())

            # manifest printout
            manifest_rows.append(
                {
                    "meta_index": int(row.meta_index),
                    "id": str(row.id),
                    "render_directory": str(row.render_directory),
                    "seg_label_source": str(row.seg_label_source),
                }
            )


    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(OUT_MANIFEST, index=False)

    print(f"Wrote TFRecord: {OUT_TFRECORD}")
    print(f"Wrote manifest: {OUT_MANIFEST}")
    print(f"Examples written: {len(manifest_df)}")


if __name__ == "__main__":
    main()