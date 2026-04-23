This subdirectory contains the Tier 3 follow-up to earlier Tier 1 / Tier 2 hemorrhage-classification work, which is not included here. Earlier tiers compared classical and ensemble models on handcrafted features; Tier 3 instead trains neural models directly on image tensors while keeping the dataset definition and split logic aligned for comparison.

This subdirectory assumes that it sits next to its sibling `/data` directory, ie.

```
project_root/
├── data/
└── tier3-neural-networks/
```

Tier 3 reads an already-built TFRecord dataset from `/data` (it does not rebuild raw data). The TFRecord contains the 4-channel input tensor, classification labels, segmentation mask, and metadata keyed by meta_index. The manifest is the companion table mainly used for split bookkeeping in this workflow.

# What is here

`scripts/models/` contains exportable keras model builders for:
- mlp.py: non-spatial baseline
- cnn.py: full-image classifier
- multiunet.py: multitask segmentation + classification model
- unet.py: segmentation-only U-Net used by Attention-CNN

`scripts/train_*.py` trains each model family and save frozen weights plus training summaries.

`scripts/input_prep.py` validates or creates train/val/test split files keyed by meta_index.

`scripts/calibrate_thresholds.py` does per-label threshold calibration on the validation split.

`scripts/evaluate_all.py` runs final evaluation on the test split and writes summary outputs.

`rules/` and `Snakefile` define the Snakemake workflow for the full Tier 3 pipeline.

`runs/<run_name>/artifacts/` receives intermediate outputs such as split summaries, model weights, calibration files, and logs.

`runs/<run_name>/results/` receives final evaluation tables and summaries.

# Workflow
- prepare or validate split files
- train MLP, CNN, MultiU-Net, and Attention-CNN
- freeze weights
- calibrate classification thresholds on validation
- evaluate all models on test

This is meant to be comparable in spirit to work done for the earlier tiers: same dataset definition, same splits, same calibration steps where possible (/ where it makes sense for neural networks). There is some modularity to easily deviate from this for exploration/testing.

# Running

From inside `tier3-neural-networks/`, most minimally, run:

`snakemake --cores 1 --resources gpu=1 all`

The workflow marks training, calibration, and evaluation jobs as using one GPU. Passing `--resources gpu=1` prevents Snakemake from running two TensorFlow jobs on the same GPU at once.

`config.yaml` controls the run folder via `run_name`. A full run writes to:

```
runs/<run_name>/
├── splits/
├── artifacts/
└── results/
```

For a new run name, `scripts/input_prep.py` initializes `runs/<run_name>/splits/` from the canonical split handoff in `splits/`, validates those copied indices, and then all downstream steps use the run-local split files.

(helpful: https://snakemake.readthedocs.io/en/stable/executing/cli.html)

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
A "vanilla NN" non-spatial baseline that is intentionally weak on spatial reasoning. The input image tensor is downsampled, flattened, and passed through a dense stack to produce 5 sigmoid classification outputs.

## CNN (Convolutional Neural Network)
A standard image classifier operating on the full 512 x 512 x 4 tensor. It uses repeated convolution and pooling blocks, followed by global pooling and a dense classification head with 5 sigmoid outputs. This is the main pure classification model for learning spatial patterns directly from the images.

## Multitask U-Net
A multitask model that performs both segmentation and classification. An encoder produces late feature maps, a decoder predicts a soft segmentation "attention" mask, and the classification head uses both globally pooled features and segmentation-weighted pooled features. It outputs 5 classification probabilities plus a 1-channel segmentation mask. This model is intended to test whether segmentation-informed features help classification of hemorrhage types.

## Attention-CNN
A two-stage model family. First, a segmentation-only U-Net is trained and its predicted soft mask is frozen. Then that mask is applied to the image tensor as `x * (attention_floor + (1 - attention_floor) * mask)`, and a CNN is trained on the masked input. In `config.yaml`, `attentioncnn.segmenter` is initialized to match the Multi-U-Net segmentation settings and `attentioncnn.classifier` is initialized to match the standalone CNN settings, but those values are independent so a later run can deliberately diverge.

# General development directions

Arranged roughly from "code refinement" to "conceptual extension"

- make outputs for different config sets saveable with names (simple refactoring)
- better documentation and computing environment specs
- allow toggle for separate pure segmentation fitting (U-Net)
- more figure printouts, including visual representations from segmentation steps
- also compute trivial baselines for comparison
- loss weight tuning for Multitask U-Net
- more systematic experimental design/evaluation scaffolding
- revisit classification chains (T1/T2) conceptually in a NN context
- other hyperparam tuning (should mostly be kept distinct from "Tier 3" work for comparability with T1/T2)
- make segmentation for Multitask U-Net a more delibeerate intermediate objective by splitting the training into two phases
