# ZooLake grouped validation study

Fixed split SHA-256 `9d0e0e2661ab8bd5f5eca506cb75049566931d31e14124a24243e3bb5e511124`. This study uses 2,690 validation images across 35 labels.

The 9 labels in the minority cohort are `bosmina`, `chaoborus`, `copepod_skins`, `daphnia_skins`, `diatom_chain`, `hydra`, `keratella_cochlearis`, `polyarthra`, `unknown_plankton`.

## Mean scores by condition

Each value is the mean and sample standard deviation across the listed seeds. Single runs have an undefined standard deviation. Seed variation describes this fixed split.

| Model | Train % | Train augmentation | Inference | Seeds (n) | Macro-F1 | Minority macro-F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| efficientnet-b0 | 25 | d4 | d4 | 464,465,466 (3) | 0.495 ± 0.092 | 0.208 ± 0.082 | 0.615 ± 0.135 |
| efficientnet-b0 | 25 | d4 | single | 464,465,466 (3) | 0.454 ± 0.082 | 0.174 ± 0.068 | 0.569 ± 0.116 |
| efficientnet-b0 | 25 | none | d4 | 464,465,466 (3) | 0.583 ± 0.042 | 0.209 ± 0.058 | 0.779 ± 0.038 |
| efficientnet-b0 | 25 | none | single | 464,465,466 (3) | 0.547 ± 0.026 | 0.186 ± 0.038 | 0.736 ± 0.035 |
| efficientnet-b0 | 100 | d4 | d4 | 464,465,466 (3) | 0.760 ± 0.019 | 0.518 ± 0.082 | 0.877 ± 0.012 |
| efficientnet-b0 | 100 | d4 | single | 464,465,466 (3) | 0.725 ± 0.014 | 0.447 ± 0.054 | 0.856 ± 0.014 |
| efficientnet-b0 | 100 | none | d4 | 464,465,466 (3) | 0.753 ± 0.014 | 0.519 ± 0.028 | 0.876 ± 0.008 |
| efficientnet-b0 | 100 | none | single | 464,465,466 (3) | 0.710 ± 0.016 | 0.437 ± 0.036 | 0.852 ± 0.004 |

## Matched D4 contrasts

Training D4 subtracts the unaugmented model at the same inference view. TTA subtracts single-view inference from averaged D4 inference on the same fitted model. Only the listed pairs contribute to the deltas. Positive values favor D4.

| Contrast | Model | Train % | Held fixed | Pairs (n) | Baseline only | D4 only | Δ macro-F1 | Δ minority macro-F1 | Δ accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d4_training | efficientnet-b0 | 25 | inference_view = single | 464,465,466 (3) | – | – | -0.092 ± 0.094 | -0.012 ± 0.096 | -0.167 ± 0.108 |
| d4_training | efficientnet-b0 | 25 | inference_view = d4 | 464,465,466 (3) | – | – | -0.088 ± 0.110 | -0.001 ± 0.103 | -0.163 ± 0.122 |
| d4_tta | efficientnet-b0 | 25 | train_augmentation = none | 464,465,466 (3) | – | – | +0.036 ± 0.017 | +0.022 ± 0.030 | +0.042 ± 0.004 |
| d4_tta | efficientnet-b0 | 25 | train_augmentation = d4 | 464,465,466 (3) | – | – | +0.040 ± 0.012 | +0.033 ± 0.017 | +0.046 ± 0.021 |
| d4_training | efficientnet-b0 | 100 | inference_view = single | 464,465,466 (3) | – | – | +0.015 ± 0.018 | +0.010 ± 0.066 | +0.004 ± 0.016 |
| d4_training | efficientnet-b0 | 100 | inference_view = d4 | 464,465,466 (3) | – | – | +0.007 ± 0.033 | -0.001 ± 0.110 | +0.001 ± 0.020 |
| d4_tta | efficientnet-b0 | 100 | train_augmentation = none | 464,465,466 (3) | – | – | +0.043 ± 0.014 | +0.082 ± 0.037 | +0.024 ± 0.005 |
| d4_tta | efficientnet-b0 | 100 | train_augmentation = d4 | 464,465,466 (3) | – | – | +0.034 ± 0.009 | +0.071 ± 0.034 | +0.021 ± 0.003 |

## Individual seeds

| Model | Train % | Train augmentation | Inference | Seed | Training images | Macro-F1 | Minority macro-F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| efficientnet-b0 | 25 | d4 | d4 | 464 | 3139 | 0.501 | 0.272 | 0.608 |
| efficientnet-b0 | 25 | d4 | d4 | 465 | 3139 | 0.399 | 0.115 | 0.485 |
| efficientnet-b0 | 25 | d4 | d4 | 466 | 3139 | 0.584 | 0.237 | 0.754 |
| efficientnet-b0 | 25 | d4 | single | 464 | 3139 | 0.454 | 0.220 | 0.553 |
| efficientnet-b0 | 25 | d4 | single | 465 | 3139 | 0.373 | 0.096 | 0.463 |
| efficientnet-b0 | 25 | d4 | single | 466 | 3139 | 0.536 | 0.207 | 0.692 |
| efficientnet-b0 | 25 | none | d4 | 464 | 3139 | 0.630 | 0.256 | 0.817 |
| efficientnet-b0 | 25 | none | d4 | 465 | 3139 | 0.572 | 0.226 | 0.740 |
| efficientnet-b0 | 25 | none | d4 | 466 | 3139 | 0.547 | 0.144 | 0.779 |
| efficientnet-b0 | 25 | none | single | 464 | 3139 | 0.574 | 0.199 | 0.772 |
| efficientnet-b0 | 25 | none | single | 465 | 3139 | 0.541 | 0.215 | 0.702 |
| efficientnet-b0 | 25 | none | single | 466 | 3139 | 0.524 | 0.144 | 0.735 |
| efficientnet-b0 | 100 | d4 | d4 | 464 | 12556 | 0.740 | 0.431 | 0.865 |
| efficientnet-b0 | 100 | d4 | d4 | 465 | 12556 | 0.759 | 0.528 | 0.878 |
| efficientnet-b0 | 100 | d4 | d4 | 466 | 12556 | 0.779 | 0.595 | 0.888 |
| efficientnet-b0 | 100 | d4 | single | 464 | 12556 | 0.716 | 0.399 | 0.843 |
| efficientnet-b0 | 100 | d4 | single | 465 | 12556 | 0.719 | 0.436 | 0.854 |
| efficientnet-b0 | 100 | d4 | single | 466 | 12556 | 0.741 | 0.506 | 0.870 |
| efficientnet-b0 | 100 | none | d4 | 464 | 12556 | 0.769 | 0.549 | 0.886 |
| efficientnet-b0 | 100 | none | d4 | 465 | 12556 | 0.745 | 0.514 | 0.872 |
| efficientnet-b0 | 100 | none | d4 | 466 | 12556 | 0.744 | 0.494 | 0.871 |
| efficientnet-b0 | 100 | none | single | 464 | 12556 | 0.721 | 0.464 | 0.856 |
| efficientnet-b0 | 100 | none | single | 465 | 12556 | 0.692 | 0.397 | 0.849 |
| efficientnet-b0 | 100 | none | single | 466 | 12556 | 0.717 | 0.450 | 0.851 |

[Learning curves](learning_curves.pdf) · [Grouped CSV](grouped_scores.csv) · [Per-seed contrasts](paired_deltas.csv) · [Pair coverage](paired_summary.csv)
