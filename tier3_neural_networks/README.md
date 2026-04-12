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

`scripts/train_*.py` trains each model family and save frozen weights plus training summaries.

`scripts/input_prep.py` validates or creates train/val/test split files keyed by meta_index.

`scripts/calibrate_thresholds.py` does per-label threshold calibration on the validation split.

`scripts/evaluate_all.py` runs final evaluation on the test split and writes summary outputs.

`rules/` and `Snakefile` define the Snakemake workflow for the full Tier 3 pipeline.

`artifacts/` receives intermediate outputs such as split files, model weights, calibration files, and logs.

`results/` receives final evaluation tables and summaries.

# Workflow
- prepare or validate split files
- train MLP, CNN, and MultiU-Net
- freeze weights
- calibrate classification thresholds on validation
- evaluate all models on test

This is meant to be comparable in spirit to work done for the earlier tiers: same dataset definition, same splits, same calibration steps where possible (/ where it makes sense for neural networks). There is some modularity to easily deviate from this for exploration/testing.

# Running

From inside `tier3-neural-networks/`, most minimally, run:

`snakemake all`

(helpful: https://snakemake.readthedocs.io/en/stable/executing/cli.html)

# Main outputs
- `artifacts/models/...` : trained model weights and training summaries
- `artifacts/calibration/...` : calibrated thresholds and calibration summaries
- `results/test_metrics_all_models.csv`
- `results/test_predictions_all_models.csv`
- `results/segmentation_metrics_multiunet.csv`
-  `results/evaluation_summary.json`

# Models

## MLP (Multilayer Perceptron)
A "vanilla NN" non-spatial baseline that is intentionally weak on spatial reasoning. The input image tensor is downsampled, flattened, and passed through a dense stack to produce 5 sigmoid classification outputs.

## CNN (Convolutional Neural Network)
A standard image classifier operating on the full 512 x 512 x 4 tensor. It uses repeated convolution and pooling blocks, followed by global pooling and a dense classification head with 5 sigmoid outputs. This is the main pure classification model for learning spatial patterns directly from the images.

## Multitask U-Net
A multitask model that performs both segmentation and classification. An encoder produces late feature maps, a decoder predicts a soft segmentation "attention" mask, and the classification head uses both globally pooled features and segmentation-weighted pooled features. It outputs 5 classification probabilities plus a 1-channel segmentation mask. This model is intended to test whether segmentation-informed features help classification of hemorrhage types.

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