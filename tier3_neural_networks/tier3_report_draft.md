This report provides a summary of prior Tier 1 / Tier 2 hemorrhage-classification experiments and the current Tier 3 neural-network workflow. The study concerns multi-label intracranial hemorrhage subtype detection from brain CT slices. The five modeled labels are epidural, intraparenchymal, intraventricular, subarachnoid, and subdural.

## Data and Cohort

The usable cohort comprised `2,929` hemorrhage-positive CT slices. Each case was represented as a `512 x 512 x 4` tensor formed from four CT window settings:

- brain-bone
- brain
- max-contrast
- subdural

Each case also has:

- five hemorrhage subtype labels
- a binary segmentation mask derived from polygon annotations
- case-level metadata

The cohort was constructed by retaining only cases with all four rendered windows, a valid multi-label classification record, and at least one usable segmentation annotation. The `any` hemorrhage label was excluded from modeling because it is uniformly positive in this hemorrhage-only cohort.

The prior Tier 1 / Tier 2 study used a `2049 / 440 / 440` train/validation/test split. The current Tier 3 run used the same cohort and the same stratified split method, but not the same subsets, producing `2050 / 439 / 440` train/validation/test cases and stratifying by true classification labels. 

## Prior Tier 1 / Tier 2 Configuration

Prior Tier 1 / Tier 2 models did not operate on raw image tensors. Each case was converted into a fixed 269-dimensional handcrafted feature vector designed to summarize intensity, texture, and coarse spatial structure.

All prior models were trained in a one-vs-rest multi-label framework, with one binary classifier per hemorrhage subtype. Features were standardized using a `StandardScaler` fit on training data only. PCA was applied to models with stronger dimensionality constraints. Hyperparameters were tuned with `RandomizedSearchCV` using 20 iterations and 5-fold stratified cross-validation, with macro AUC-ROC as the primary selection metric.

Two post-training procedures were used in the prior study. First, each model underwent per-label threshold calibration on the validation split. Thresholds were swept from `0.05` to `0.95` in increments of `0.01`, and the threshold maximizing binary F1 was selected independently for each label. Second, classifier chains were evaluated as a possible label-dependence mechanism. In practice, they did not improve performance: reported AUC changes were within 0.006, effectively indistinguishable from noise.

The main T1/T2 conclusions were:
- all Tier 2 models outperformed all Tier 1 models on test AUC,
- Random Forest achieved the highest test macro AUC (0.744),
- XGBoost achieved the highest test macro AUPRC (0.469),
- Logistic Regression was the strongest Tier 1 model (0.697 AUC, 0.410 AUPRC).
- Intraventricular hemorrhage was the easiest label despite low prevalence,
- Subarachnoid and subdural hemorrhages were the hardest labels across models.
- Threshold calibration materially improved performance, especially for models without class weighting.

## Tier 3 

The current Tier 3 workflow reads the existing TFRecord dataset and uses the same five classification labels as the prior tiers. Unlike Tier 1 / Tier 2, Tier 3 trains directly on the 512 x 512 x 4 image tensor and can also consume the segmentation mask as a supervision target.

The current workflow evaluates three neural model families.

| Model | Input handling | Architecture | Batch size | Loss |
| --- | --- | --- | ---: | --- |
| MLP | Resize to `64 x 64`, then flatten | Dense layers `[512, 256]`, batch norm, ReLU, dropout `0.3`, 5-way sigmoid output | 4 | Binary cross-entropy |
| CNN | Full `512 x 512 x 4` tensor | 5 conv blocks with filters `[8, 16, 32, 64, 128]`, kernel size `3`, global average pooling, dense layer `[256]`, batch norm, dropout `0.3`, 5-way sigmoid output | 4 | Binary cross-entropy |
| MultiU-Net | Full `512 x 512 x 4` tensor | Encoder `[16, 32, 64, 128]`, bottleneck `256`, decoder `[128, 64, 32, 16]`, segmentation head plus classification head `[256]` using pooled and segmentation-weighted features | 2 | Binary cross-entropy + Dice loss |

Shared training settings in the current workflow were:

- optimizer: Adam
- learning rate: `1e-4`
- maximum epochs: `50`
- early stopping patience: `10`

For the MultiU-Net, classification and segmentation losses were equally weighted:

- classification loss weight: `0.5`
- segmentation loss weight: `0.5`

As in the prior tiers, each Tier 3 model underwent per-label threshold calibration on the validation split using a sweep from `0.05` to `0.95` in increments of `0.01`. Final test predictions used the calibrated thresholds.

Classification evaluation used macro AUC-ROC, macro AUPRC, macro F1, and per-label AUC/AUPRC/F1. For the MultiU-Net, mean Dice score at segmentation threshold 0.5 was also reported.

## Tier 3 Results

### Calibration

Validation-set macro F1 before and after threshold calibration was:

| Model | Default F1 | Calibrated F1 | Gain |
| --- | ---: | ---: | ---: |
| MLP | 0.235 | 0.459 | +0.224 |
| CNN | 0.214 | 0.477 | +0.262 |
| MultiU-Net | 0.204 | 0.502 | +0.298 |

Threshold calibration again produced large gains, consistent with the prior tiers.

### Macro-averaged Tier-3 Results

Macro-averaged Tier 3 test results were:

| Model | Test AUC | Test AUPRC | Test F1 |
| --- | ---: | ---: | ---: |
| MLP | 0.635 | 0.349 | 0.388 |
| CNN | 0.735 | 0.506 | 0.490 |
| MultiU-Net | 0.744 | 0.471 | 0.485 |

The principal patterns are straightforward:

- the non-spatial MLP underperformed both spatial neural models
- the CNN achieved the strongest macro AUPRC (`0.506`) and macro F1 (`0.490`)
- the MultiU-Net achieved the highest macro AUC (`0.744`) while also producing segmentation output

The AUC-to-AUPRC gaps remained substantial. Thus, direct image modeling improved performance in important respects, but did not remove the precision burden imposed by imbalance.

### Per-Label Tier 3 Results

Per-label Tier 3 test AUC values were:

| Label | MLP | CNN | MultiU-Net |
| --- | ---: | ---: | ---: |
| Epidural | 0.675 | 0.823 | 0.797 |
| Intraparenchymal | 0.653 | 0.730 | 0.790 |
| Intraventricular | 0.667 | 0.803 | 0.847 |
| Subarachnoid | 0.538 | 0.635 | 0.615 |
| Subdural | 0.644 | 0.683 | 0.673 |

The strongest Tier 3 model by label was therefore:

- CNN for epidural, subarachnoid, and subdural
- MultiU-Net for intraparenchymal and intraventricular

This preserves one of the key prior qualitative findings: subarachnoid hemorrhage remains the most difficult label, whereas intraventricular hemorrhage remains one of the most detectable.

### Segmentation Result

The MultiU-Net achieved a mean Dice score of `0.480` at segmentation threshold `0.5`. This metric has no direct Tier 1 / Tier 2 analogue, but it indicates that the multitask architecture learned a non-trivial segmentation signal while maintaining competitive classification performance.

## Provisional Cross-Tier Comparison

The table below juxtaposes the strongest prior models with the current Tier 3 models.

| Group | Model | AUC | AUPRC | F1 |
| --- | --- | ---: | ---: | ---: |
| Prior Tier 1 best | Logistic Regression | 0.697 | 0.410 | 0.441 |
| Prior Tier 2 best AUC/F1 | Random Forest | 0.744 | 0.469 | 0.475 |
| Prior Tier 2 best AUPRC | XGBoost | 0.731 | 0.469 | 0.462 |
| Tier 3 | MLP | 0.635 | 0.349 | 0.388 |
| Tier 3 | CNN | 0.735 | 0.506 | 0.490 |
| Tier 3 | MultiU-Net | 0.744 | 0.471 | 0.485 |

Three provisional conclusions folow:

- Neural parameterization alone is not sufficient. The MLP, which discards most spatial structure before learning, is weaker than the strongest prior classical and ensemble baselines.
- Direct spatial modeling appears beneficial. On the current split, the CNN exceeds the strongest prior AUPRC and F1, while the MultiU-Net matches the strongest prior AUC and slightly exceeds the strongest prior AUPRC.
- The same label-level bottleneck remains visible across tiers. Subarachnoid hemorrhage remains difficult even under direct image modeling, suggesting that the move from handcrafted features to neural architectures improves but does not eliminate the core surface-hemorrhage failure mode identified in prior work.

## Conclusion

Prior Tier 1 / Tier 2 work established a strong handcrafted-feature baseline for hemorrhage subtype classification. Tier 2 ensemble methods improved substantially over Tier 1 statistical models, with the largest gains on anatomically diffuse labels, but subarachnoid and subdural hemorrhages remained difficult. Threshold calibration was essential throughout. The current Tier 3 workflow evaluates three neural models trained directly on image tensors. The results indicate that spatial modeling is critical: the MLP underperforms, whereas the CNN and MultiU-Net are competitive with the strongest prior baselines and show improved precision-recall behavior on the current split. However, training on the hardest morphology regimes identified previously, especially subarachnoid hemorrhage, remains difficult.

