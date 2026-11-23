# Scratch CNN V2 - Experiment Results

## Overview

This folder contains the results of the Scratch CNN V2 experiment for the CS464 ZooLake classification project.

The experiment evaluates the effect of training data quantity and D4 data augmentation on classification performance.

## Experiment Design

- Model: Scratch CNN V2
- Training fractions: 10%, 25%, 50%, 100%
- Random seeds: 464, 465, 466
- Training augmentation: None and D4
- Total runs: 24
- Image size: 96 x 96
- Batch size: 64
- Optimizer: AdamW
- Initial learning rate: 0.001
- Weight decay: 0.0001
- Learning rate scheduler: Cosine Annealing
- Maximum epochs: 50
- Early stopping patience: 6
- Class weighting: Balanced
- Device: CUDA

## Dataset and Split

The experiment uses the frozen group-based split.

- Training images: 12,556
- Validation images: 2,690
- Test images: 2,691
- Number of classes: 35

Split manifest SHA-256:

`cffff06a321d2949f7298d71343584e33883d4e2db7bb600125f93cbcfb1e2a6`

The same split manifest was used for all 24 runs.

## Evaluation

The best checkpoint for each run was selected using validation macro-F1.

Two prediction views were evaluated:

- Single: normal single-view prediction.
- D4: test-time prediction using D4 transformations.

Training augmentation and D4 inference are separate experimental settings.

The reported results are validation results. The test set has not been used for model selection.

## Aggregate Results

Mean and sample standard deviation across three seeds:

| Training Fraction | Augmentation | Macro-F1 | Minority Macro-F1 | Accuracy |
|---|---|---:|---:|---:|
| 10% | D4 | 0.2819 ± 0.0128 | 0.0936 ± 0.0404 | 0.4450 ± 0.0310 |
| 10% | None | 0.2932 ± 0.0095 | 0.0783 ± 0.0370 | 0.4535 ± 0.0056 |
| 25% | D4 | 0.3535 ± 0.0561 | 0.1369 ± 0.0603 | 0.5113 ± 0.0299 |
| 25% | D4 | 0.3535 ± 0.0561 | 0.1369 ± 0.0603 | 0.5113 ± 0.0299 |
| 25% | None | 0.3819 ± 0.0343 | 0.1483 ± 0.0373 | 0.5388 ± 0.0294 |
| 50% | D4 | 0.4985 ± 0.0203 | 0.2472 ± 0.0396 | 0.6502 ± 0.0323 |
| 50% | None | 0.4997 ± 0.0126 | 0.2356 ± 0.0095 | 0.6608 ± 0.0127 |
| 100% | D4 | 0.6058 ± 0.0269 | 0.3531 ± 0.0413 | 0.7487 ± 0.0228 |
| 100% | None | 0.6090 ± 0.0055 | 0.3599 ± 0.0158 | 0.7545 ± 0.0071 |

## Main Observations

- Increasing the training fraction substantially improves macro-F1 and minority macro-F1.
- D4 training augmentation does not provide a consistent improvement over training without augmentation.
- The 100% training fraction without augmentation achieves the highest average validation macro-F1, minority macro-F1, and accuracy.
- Results at smaller training fractions show greater variation across random seeds.

## Folder Structure

Each run has its own directory. The directory name identifies its seed, training fraction, and augmentation setting.

Each run directory contains:

- `metrics.json`: evaluation metrics for single and D4 prediction views.
- `training_log.json`: epoch-level training and validation history.
- `run.json`: experiment configuration and provenance.
- `predictions.csv`: validation predictions.
- `per_class_f1.png`: per-class F1 visualization.

Model checkpoint files are not included in this results folder on GitHub.

## Reproducibility

The training implementation and configuration files are available in:

- `zoolake-grouped/zoolake/cnn_v2.py`
- `zoolake-grouped/scripts/run_scratch_v2.py`
- `zoolake-grouped/configs/scratch_v2_final.toml`

All runs use the same frozen split manifest and record their configuration and checkpoint hash in `run.json`.