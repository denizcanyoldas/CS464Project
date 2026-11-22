# Grouped ZooLake final study

The study uses one fixed, capture-grouped holdout. On the **2,691-image final test**, the selected EfficientNet-B0 scored **0.7800 macro-F1**, **0.5779 minority macro-F1**, and **90.00% accuracy**. The choice was frozen using validation scores from full-budget seed-464 models. Its six-epoch ImageNet-initialized EfficientNet-B0 was trained without geometric augmentation and scored with mean probabilities over the eight D4 views. [Frozen selection](selection.json) · [Test metrics](test/test_metrics.json)

## What was measured

- The original Eawag release contains 17,943 images and 35 labels. After removing six files from three conflicting exact duplicate pairs, the three-minute UTC grouped split assigns 12,556 train, 2,690 validation, and 2,691 test images. Every label appears in each partition. The 10/25/50/100% nested budgets contain 1,256 / 3,139 / 6,278 / 12,556 original training images. [Split summary](../split_summary.md) · [Portable split identity](test/split_identity.json)
- The main comparison contains 35 fits, including the earlier LR and two new CPU baselines. Scratch CNN uses three seeds across all four budgets and two augmentation arms. ImageNet-pretrained EfficientNet-B0 uses one seed over the same four budgets and two arms. Two further scratch fits isolate cross-entropy weighting at 10% and 100%, making 37 fits in total. The 36 new runs ran for about 37.8 minutes of summed fit and canonical validation prediction time. [Study matrix](../../configs/final_study.toml) · [Individual runs](validation/individual_scores.csv)
- CNNs use 96-pixel RGB inputs, AdamW with learning rate 0.001, batches of 64, and six epochs. Training batches use inverse-frequency cross-entropy weights within the main comparison. Within each D4 pair, the same seed and nested training images are used. D4 applies one of four quarter turns with or without reflection per image and epoch. Each checkpoint is chosen by single-view validation macro-F1. Classical models use the 104-feature 128-pixel pipeline from the progress report.

## Validation findings

The table holds inference at one view to isolate the training augmentation setting. Scratch reports mean ± sample SD across seeds 464–466. EfficientNet-B0 uses seed 464.

| Training images | Scratch plain | Scratch D4 | EfficientNet plain | EfficientNet D4 |
| --- | ---: | ---: | ---: | ---: |
| 1,256 (10%) | 0.040 ± 0.004 | 0.035 ± 0.015 | 0.401 | 0.466 |
| 3,139 (25%) | 0.125 ± 0.017 | 0.103 ± 0.028 | 0.574 | 0.454 |
| 6,278 (50%) | 0.221 ± 0.014 | 0.208 ± 0.014 | 0.593 | 0.653 |
| 12,556 (100%) | 0.328 ± 0.019 | 0.337 ± 0.017 | 0.721 | 0.716 |

More training images clearly help the scratch CNN under the fixed six-epoch budget. The number of optimizer updates also grows with the dataset. The EfficientNet D4 training effect varies by budget in the single-seed comparison. At 25% of the data, plain EfficientNet with one view scores 0.5745, above the three classical full-budget baselines on this split. Those baselines score 0.5122 for LR, 0.5133 for RBF-SVM and 0.5289 for XGBoost. A classifier that always predicts the most frequent training label, `dinobryon`, scores 0.0104 macro-F1 and 22.12% validation accuracy. [All scores and pair coverage](validation/summary.md) · [Learning curves](validation/learning_curves.pdf)

Inference averaging helped overall macro-F1 in all eight EfficientNet conditions at seed 464. For the selected full-budget checkpoint, it moves macro-F1 from **0.7212 to 0.7692** and minority macro-F1 from **0.4640 to 0.5489**. The two canonical inference modes agree on 90.04% of validation predictions. Across 2,690 images, model forward and probability return took 1.31 s for one view and 10.35 s for D4, about 7.9× the single-view time on Metal. The scratch CNN showed much smaller and less consistent TTA changes. [Paired effects](validation/paired_summary.csv)

For scratch with D4 training and one view at 100%, removing the class weights raises overall F1 from 0.3180 to 0.3795 and drops minority F1 from 0.1300 to 0. At 10%, weighted and unweighted minority F1 are 0.0042 and 0 respectively. These loss comparisons use one matched seed. [Loss weighting rows](weighting.csv)

## Rotation and test diagnostics

The validation angle sweep uses eight fixed angles from 0° to 315° in 45° increments. Each ROI reuses one canvas with enough clearance for all angles. Quarter turns permute pixels and diagonal views use bicubic interpolation. The angle sweep's 0° preprocessing can differ from canonical validation for oversized ROIs.

For the selected EfficientNet checkpoint, one-view mean/worst angle macro-F1 are **0.6099 / 0.5812**. D4 inference raises these to **0.7008 / 0.6910**. The selected checkpoint without training augmentation also has a better D4 angle mean than the D4-trained checkpoint, which scores 0.6581. Under one-view inference the D4-trained checkpoint has the higher mean angle score, 0.6217. Agreement with the selected checkpoint's 0° prediction averages 0.7612 for one view and 0.9420 for D4. The identical quarter-turn D4 results follow from using the same eight-view orbit. [Rotation scores](rotation_summary.csv) · [Selected model angle plot](rotations/transfer-roi-f100-none-s464/rotations.png)

The largest test confusions by count are `unknown → rotifers` (16) and `eudiaptomus → cyclops` (15). Minority estimates are uncertain at the smallest supports. For example, `chaoborus` has two test examples and `hydra` has three. Folds may share longer capture sessions across three-minute boundaries. The separate visual screen found zero candidates at its primary near-duplicate threshold and 610 visually similar pairs at a looser threshold in train versus validation. Those candidates are visually similar images with unknown capture identity. [Class scores and supports](test/test_per_class.csv) · [Test confusion plot](test/test_confusion.pdf) · [Visual screening rules and counts](../near_duplicate_screen.json)

## Reproduce

From the grouped project directory, prepare the dataset and environment using the [main README](../../README.md). Run the classifier commands from [the study matrix](../../configs/final_study.toml) with an available accelerator. Full reruns need the raw data. The checkpoints and per-image predictions are generated under the local `artifacts/` directory.

```bash
# Generate this path on a fresh checkout after audit and split.
python -m zoolake train --out artifacts/runs/baseline-lr-s464
python scripts/run_final_study.py scratch transfer weighting
python scripts/run_final_study.py svm
# Configure OpenMP as described in the main README if XGBoost requires it.
python scripts/run_final_study.py xgb

python -m zoolake.study --runs artifacts/runs/baseline-lr-s464 \
  artifacts/runs/final-study/{scratch-*,transfer-*,svm-*,xgb-*} --out artifacts/study-replay
python -m zoolake.rotations --run artifacts/runs/final-study/transfer-efficientnet-b0-f100-none-s464 \
  --tta --device mps --out artifacts/rotations/replay
```

The [frozen selection](selection.json) identifies the recorded checkpoint, split and inference policy. To reproduce the final-test protocol on a fresh set of local artifacts, first freeze the listed validation candidate families from the reproduced run directories. Keep all selection decisions on validation.

```bash
python -m zoolake.finalize freeze --runs artifacts/runs/baseline-lr-s464 \
  artifacts/runs/final-study/*f100*s464 --split artifacts/splits/splits.json \
  --out artifacts/replay-selection.json
python -m zoolake.finalize evaluate --selection artifacts/replay-selection.json \
  --split artifacts/splits/splits.json --data-root ../data/zoolake --out artifacts/replay-test
python scripts/export_test_results.py --test artifacts/replay-test \
  --split artifacts/splits/splits.json --out artifacts/replay-test-figures
```

Run `python -m zoolake.predict --run artifacts/runs/final-study/transfer-efficientnet-b0-f100-none-s464 --image <local-image.jpg> --view d4` for a top-k inference demo with the selected checkpoint.
