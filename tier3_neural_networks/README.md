This subdirectory contains the Tier 3 follow-up to earlier Tier 1 / Tier 2 hemorrhage-classification work, which is not included here. Earlier tiers compared classical and ensemble models on handcrafted features; Tier 3 instead trains neural models directly on image tensors while keeping the dataset definition, split logic, and threshold-calibration protocol aligned for comparison.

This subdirectory assumes that it sits next to its sibling `/data` directory, ie.

```
project_root/
├── data/
└── tier3-neural-networks/
```

Tier 3 reads an already-built TFRecord dataset from `/data` (it does not rebuild raw data). The TFRecord contains the 4-channel input tensor, classification labels, segmentation mask, and metadata keyed by `meta_index`. The manifest is the companion table mainly used for split validation in this workflow.

# What is here

This GitHub subdirectory currently contains:

- `Snakefile`: the top-level Snakemake entry point
- `config.yaml`: model, training, path, and split settings
- `env.yaml`: the conda environment definition for the main reported work
- `README.md`: this overview
- `final_report.md`: the rendered GitHub Markdown report
- `.gitignore`: local ignore rules for generated artifacts
- `rules/`: Snakemake rule fragments for input prep, training, calibration, and evaluation
- `scripts/`: Python code for data loading, model building, training, calibration, and inference

Within `scripts/`, the most relevant pieces are `train_mlp.py`, `train_cnn.py`, `train_multiunet.py`, `train_attentioncnn.py`, `input_prep.py`, `calibrate_thresholds.py`, `evaluate_all.py`, and the shared helpers under `scripts/shared/`. The model builders live under `scripts/models/`.

# Workflow
- prepare or validate split files
- train MLP, CNN, and Multi-U-Net
- calibrate classification thresholds on validation
- evaluate all models on test
- optionally train and evaluate Attention-CNN by extending `config.workflow.model_families`

This is meant to be comparable in spirit to work done for the earlier tiers: same dataset definition, same splits, same calibration steps where they make sense for neural networks, and the same held-out test discipline.

# Running

From inside `tier3_neural_networks/`, most minimally, run:

`snakemake --cores 1 --resources gpu=1 all`

`config.yaml` controls the run folder via `run_name`. A full run writes to:

```
runs/<run_name>/
├── splits/
├── artifacts/
└── results/
```

(helpful: https://snakemake.readthedocs.io/en/stable/executing/cli.html)

For a new run name, `scripts/input_prep.py` initializes `runs/<run_name>/splits/` from the canonical split handoff in `splits/` if present, validates those copied indices, and then all downstream steps use the run-local split files. The default `run_name` is set in `config.yaml`.

# Main outputs
- `runs/<run_name>/splits/...` : run-local train/validation/test split files
- `runs/<run_name>/artifacts/models/...` : trained model weights and training summaries
- `runs/<run_name>/artifacts/calibration/...` : calibrated thresholds and calibration summaries
- `runs/<run_name>/results/test_metrics_all_models.csv`
- `runs/<run_name>/results/test_predictions_all_models.csv`
- `runs/<run_name>/results/segmentation_metrics_all_models.csv`
-  `runs/<run_name>/results/evaluation_summary.json`

# Models

## MLP (Multilayer Perceptron)
A non-spatial baseline that downsamples the input tensor, flattens it, and passes it through a dense stack to produce 5 sigmoid classification outputs.

## CNN (Convolutional Neural Network)
A standard image classifier operating on the full 512 x 512 x 4 tensor. It uses repeated convolution and pooling blocks, followed by global pooling and a dense classification head with 5 sigmoid outputs. This is the main pure classification model for learning spatial patterns directly from the images.

## Multitask U-Net
A multitask model that performs both segmentation and classification. An encoder produces late feature maps, a decoder predicts a soft segmentation mask, and the classification head uses both globally pooled features and segmentation-weighted pooled features. It outputs 5 classification probabilities plus a 1-channel segmentation mask. This model tests whether segmentation-informed features help classification of hemorrhage types.

## Attention-CNN
A two-stage experimental model family. First, a segmentation-only U-Net is trained and its predicted soft mask is frozen. Then that mask is applied to the image tensor as `x * (attention_floor + (1 - attention_floor) * mask)`, and a CNN is trained on the masked input. In `config.yaml`, `attentioncnn.segmenter` is initialized to match the Multi-U-Net segmentation settings and `attentioncnn.classifier` is initialized to match the standalone CNN settings, but those values are independent so a later run can deliberately diverge.
