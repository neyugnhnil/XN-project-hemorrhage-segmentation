This repository was created to collaborate with team members for XN project Brain CT image hemorrhage segmentation.

This public repository currently contains three Python scripts, though they require and assume locally accessible data in `/data`.

## tl;dr/quickstart for collaborators
- all output/input files should be in `/data`
- `make_metafile.py` is how I filtered the CT scan IDs down to ones with all four CT render windows, a classification label, and a Correct or Majority segmentation polygon label. 
	- `subset_report.json` is a summary of the filtering and final metafile of usable cases
- `write_tfrecord.py` is a script that references the metafile to create a `.tfrecord` file (and associated manifest). This file stores serialized `tf.train.Example` records.
- `load_tfrecord.py` defines a function that reads those records and returns a `tf.data.Dataset`.

If you have the `cases.tfrecord` in `/data`, you do not need the raw JPGs or CSVs to load it.

```python
from load_tfrecord import load_tfrecord
import tensorflow as tf

ds = load_tfrecord("data/cases.tfrecord")
```

`ds` is the tensorflow dataset; an iterable where each step yields one batch. By default, `include_metadata=False`, so each element has structure `(x, {"cls": y_cls, "seg": y_seg})` where:

- `x[i]` is a 4-channel tensor of shape `[batch size, H, W, C] = [batch size, 512, 512, 4]` (the CT scan render windows are the 4 channels)

- `y_cls[i]` is a classification label of shape `[batch size, 6]`

- `y_seg[i]` is a segmentation label of shape `[batch size, 512, 512, 1]`; this is made by rasterizing the polygon into a binary mask where 1 = hemorrhage, 0 = background

`load_tfrecord` also batches the examples, with a default batch size of 4.

you can check that `print(ds.element_spec)` returns:

```
(TensorSpec(shape=(None, 512, 512, 4), dtype=tf.float32, name=None), {'cls': TensorSpec(shape=(None, 6), dtype=tf.float32, name=None), 'seg': TensorSpec(shape=(None, 512, 512, 1), dtype=tf.float32, name=None)})
```


## `/data`

`/data` is empty in the public repository, but should have this structure:
```
.  
├── renders  
│   ├── epidural  
│   │   ├── brain_bone_window  
│   │   ├── brain_window  
│   │   ├── max_contrast_window  
│   │   └── subdural_window  
│   ├── intraparenchymal  
│   │   ├── brain_bone_window  
│   │   ├── brain_window  
│   │   ├── max_contrast_window  
│   │   └── subdural_window  
│   ├── intraventricular  
│   │   ├── brain_bone_window  
│   │   ├── brain_window  
│   │   ├── max_contrast_window  
│   │   └── subdural_window  
│   ├── multi  
│   │   ├── brain_bone_window  
│   │   ├── brain_window  
│   │   ├── max_contrast_window  
│   │   └── subdural_window  
│   ├── normal  
│   │   ├── brain_bone_window  
│   │   ├── brain_window  
│   │   ├── max_contrast_window  
│   │   └── subdural_window   
│   ├── subarachnoid  
│   │   ├── brain_bone_window  
│   │   ├── brain_window  
│   │   ├── max_contrast_window  
│   │   └── subdural_window  
│   └── subdural  
│       ├── brain_bone_window  
│       ├── brain_window  
│       ├── max_contrast_window  
│       └── subdural_window  
└── segmentation
```

Additionally, `data/hemorrhage-labels.csv` and `data/flagged.txt` are required to exist.

Directory and file names are hardcoded in the script, but the paths can be easily changed within the scripts as they are only defined once. 

Otherwise, the structure and file naming conventions are the same as what can be retrieved directly from the data source.

## `make_metafile.py`

**dependencies**: `json`, `pandas`

`make_metafile.py` creates a `subset_metafile.csv` of CT scan IDs, classification labels, and segmentation labels from the contents of /data, based on inclusion rules. 

It accesses all raw data and identifies unique IDs meeting these criteria:

- the ID is not listed in `data/flagged.txt`

- within at least one hemorrhage type subdirectory, the ID is present across all 4 windows

- the ID has classification labels in `data/hemorrhage-labels.csv`

- the ID has at least one usable segmentation annotation in `data/segmentation/results_*.csv`

When an ID has multiple usable associated segmentation labels, one is chosen using these rules (in order of descending priority):

- prefer Correct label over Majority label

- prefer higher Agreement

- prefer higher Total Qualified Reads

- prefer higher Total Reads

- prefer earlier source CSV name

- prefer earlier Case ID

A successfully built `subset_metafile.csv` will have the following fields:

`id`: CT scan id

`render_directory`: the hemorrhage type directory the CT scan render windows belong to.

`any`, `epidural`, `intraparenchymal`, `intraventricular`, `subarachnoid`, `subdural`: classification flags from `data/hemorrhage-labels.csv`

`seg_label_source`: either `correct` or `majority`

`seg_polygon`: the segmentation label's polygon coordinates

`brain_bone_window_path`, `brain_window_path`, `max_contrast_window_path`, `subdural_window_path`: relative paths to the .jpgs for the four CT scan render windows 

The metafile creation script also outputs `subset_report.json`. An example is included in the public repository. 

(Using these inclusion rules for the raw data for this project creates a 2929-entry `subset_metafile.csv`, a significant reduction from 752803 IDs found in `data/hemorrhage-labels.csv`. This is due to the availability of segmentation label data, which exists for 2984 unique IDs only.)

## `write_tfrecord.py`

**dependencies**: `ast`, `json`, `pathlib`, `numpy`, `pandas`, `tensorflow`, `opencv-python`

`write_tfrecord.py` uses `data/subset_metafile.csv` as a reference to create a TensorFlow-native on-disk dataset that can be loaded during training. It also writes `data/cases_manifest.csv`, a lightweight index of the written examples.

For each `tf.train.Example`, the script:
- reads the four grayscale render JPGs
- rescales them to `512 x 512`
- stacks them into a 4-channel input tensor `x`
- reads the six classification labels into `y_cls`
- converts the polygon string into a binary mask `y_seg`
- serializes `x`, `y_cls`, and `y_seg` into a `tf.train.Example`

So each serialized example contains:
- metadata:
  - `meta_index`
  - `id`
  - `render_directory`
  - `seg_label_source`
  - `height`
  - `width`
  - `num_channels`
  - `num_class_labels`
- serialized tensors:
  - `x_raw`
  - `y_cls_raw`
  - `y_seg_raw` 

## `load_tfrecord.py`

**dependencies**: `tensorflow`

`load_tfrecord.py` defines rools to read a `.tfrecord` and reconstruct it as a live `tf.data.Dataset`. 
It deserializes the tensors written by `write_tfrecord.py`, restores their expected shapes, and returns them in a format that can be fed directly into a tf workflow.

For each serialized `tf.train.Example`, the script:
- parses the stored feature fields from the TFRecord
- deserializes `x_raw`, `y_cls_raw`, and `y_seg_raw` back into tensors
- restores the expected shapes:
  - `x`: `[512, 512, 4]`
  - `y_cls`: `[6]`
  - `y_seg`: `[512, 512, 1]`
- groups the targets into a dictionary `y = {"cls": y_cls, "seg": y_seg}`
- optionally includes metadata alongside the tensors

So each loaded dataset element contains either:
- `(x, y)`, where:
  - `x` is the 4-channel input image tensor
  - `y` is a dictionary with:
    - `"cls"`: the six classification labels
    - `"seg"`: the binary segmentation mask

or, if metadata is requested:
- `(x, y, metadata)`, where `metadata` contains:
  - `meta_index`
  - `id`
  - `render_directory`
  - `seg_label_source`

At the dataset level, the script can also optionally shuffle the examples and batch them into groups of size `batch_size` (default 4).

