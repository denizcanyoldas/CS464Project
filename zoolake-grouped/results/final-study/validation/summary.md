# ZooLake grouped validation study

Fixed split SHA-256 `58e6fc7fc5e1935c43f035922b273440a6609e45d6050a86e2c11a251249bc91`. This study uses 2,690 validation images across 35 labels.

The 9 labels in the minority cohort are `bosmina`, `chaoborus`, `copepod_skins`, `daphnia_skins`, `diatom_chain`, `hydra`, `keratella_cochlearis`, `polyarthra`, `unknown_plankton`.

## Mean scores by condition

Each value is the mean and sample standard deviation across the listed seeds. Single runs have an undefined standard deviation. Seed variation describes this fixed split.

| Model | Train % | Train augmentation | Inference | Seeds (n) | Macro-F1 | Minority macro-F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| efficientnet-b0 | 10 | d4 | d4 | 464 (1) | 0.495 | 0.144 | 0.674 |
| efficientnet-b0 | 10 | d4 | single | 464 (1) | 0.467 | 0.156 | 0.631 |
| efficientnet-b0 | 10 | none | d4 | 464 (1) | 0.408 | 0.113 | 0.646 |
| efficientnet-b0 | 10 | none | single | 464 (1) | 0.401 | 0.088 | 0.623 |
| efficientnet-b0 | 25 | d4 | d4 | 464 (1) | 0.501 | 0.272 | 0.608 |
| efficientnet-b0 | 25 | d4 | single | 464 (1) | 0.454 | 0.220 | 0.553 |
| efficientnet-b0 | 25 | none | d4 | 464 (1) | 0.630 | 0.256 | 0.817 |
| efficientnet-b0 | 25 | none | single | 464 (1) | 0.574 | 0.199 | 0.772 |
| efficientnet-b0 | 50 | d4 | d4 | 464 (1) | 0.690 | 0.438 | 0.832 |
| efficientnet-b0 | 50 | d4 | single | 464 (1) | 0.653 | 0.394 | 0.788 |
| efficientnet-b0 | 50 | none | d4 | 464 (1) | 0.633 | 0.313 | 0.787 |
| efficientnet-b0 | 50 | none | single | 464 (1) | 0.593 | 0.284 | 0.742 |
| efficientnet-b0 | 100 | d4 | d4 | 464 (1) | 0.740 | 0.431 | 0.865 |
| efficientnet-b0 | 100 | d4 | single | 464 (1) | 0.716 | 0.399 | 0.843 |
| efficientnet-b0 | 100 | none | d4 | 464 (1) | 0.769 | 0.549 | 0.886 |
| efficientnet-b0 | 100 | none | single | 464 (1) | 0.721 | 0.464 | 0.856 |
| lr | 100 | none | single | 464 (1) | 0.512 | 0.145 | 0.694 |
| scratch | 10 | d4 | d4 | 464,465,466 (3) | 0.036 ± 0.017 | 0.009 ± 0.015 | 0.114 ± 0.030 |
| scratch | 10 | d4 | single | 464,465,466 (3) | 0.035 ± 0.014 | 0.001 ± 0.002 | 0.114 ± 0.030 |
| scratch | 10 | none | d4 | 464,465,466 (3) | 0.040 ± 0.006 | 0.002 ± 0.004 | 0.129 ± 0.009 |
| scratch | 10 | none | single | 464,465,466 (3) | 0.040 ± 0.004 | 0.002 ± 0.004 | 0.129 ± 0.011 |
| scratch | 25 | d4 | d4 | 464,465,466 (3) | 0.104 ± 0.028 | 0.000 ± 0.000 | 0.219 ± 0.049 |
| scratch | 25 | d4 | single | 464,465,466 (3) | 0.103 ± 0.028 | 0.000 ± 0.000 | 0.218 ± 0.049 |
| scratch | 25 | none | d4 | 464,465,466 (3) | 0.122 ± 0.018 | 0.004 ± 0.008 | 0.261 ± 0.048 |
| scratch | 25 | none | single | 464,465,466 (3) | 0.125 ± 0.016 | 0.004 ± 0.007 | 0.264 ± 0.046 |
| scratch | 50 | d4 | d4 | 464,465,466 (3) | 0.206 ± 0.011 | 0.025 ± 0.008 | 0.337 ± 0.017 |
| scratch | 50 | d4 | single | 464,465,466 (3) | 0.208 ± 0.014 | 0.018 ± 0.004 | 0.341 ± 0.019 |
| scratch | 50 | none | d4 | 464,465,466 (3) | 0.217 ± 0.017 | 0.029 ± 0.022 | 0.342 ± 0.020 |
| scratch | 50 | none | single | 464,465,466 (3) | 0.221 ± 0.015 | 0.031 ± 0.026 | 0.344 ± 0.019 |
| scratch | 100 | d4 | d4 | 464,465,466 (3) | 0.336 ± 0.020 | 0.119 ± 0.025 | 0.486 ± 0.023 |
| scratch | 100 | d4 | single | 464,465,466 (3) | 0.337 ± 0.017 | 0.117 ± 0.028 | 0.487 ± 0.021 |
| scratch | 100 | none | d4 | 464,465,466 (3) | 0.325 ± 0.016 | 0.118 ± 0.050 | 0.464 ± 0.015 |
| scratch | 100 | none | single | 464,465,466 (3) | 0.328 ± 0.019 | 0.116 ± 0.055 | 0.469 ± 0.013 |
| svm | 100 | none | single | 464 (1) | 0.513 | 0.149 | 0.739 |
| xgb | 100 | none | single | 464 (1) | 0.529 | 0.130 | 0.745 |

## Matched D4 contrasts

Training D4 subtracts the unaugmented model at the same inference view. TTA subtracts single-view inference from averaged D4 inference on the same fitted model. Only the listed pairs contribute to the deltas. Positive values favor D4.

| Contrast | Model | Train % | Held fixed | Pairs (n) | Baseline only | D4 only | Δ macro-F1 | Δ minority macro-F1 | Δ accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| d4_training | efficientnet-b0 | 10 | inference_view = single | 464 (1) | – | – | +0.065 | +0.068 | +0.007 |
| d4_training | efficientnet-b0 | 10 | inference_view = d4 | 464 (1) | – | – | +0.087 | +0.031 | +0.028 |
| d4_tta | efficientnet-b0 | 10 | train_augmentation = none | 464 (1) | – | – | +0.007 | +0.025 | +0.022 |
| d4_tta | efficientnet-b0 | 10 | train_augmentation = d4 | 464 (1) | – | – | +0.028 | -0.012 | +0.043 |
| d4_training | efficientnet-b0 | 25 | inference_view = single | 464 (1) | – | – | -0.120 | +0.020 | -0.219 |
| d4_training | efficientnet-b0 | 25 | inference_view = d4 | 464 (1) | – | – | -0.129 | +0.015 | -0.209 |
| d4_tta | efficientnet-b0 | 25 | train_augmentation = none | 464 (1) | – | – | +0.055 | +0.057 | +0.045 |
| d4_tta | efficientnet-b0 | 25 | train_augmentation = d4 | 464 (1) | – | – | +0.047 | +0.052 | +0.055 |
| d4_training | efficientnet-b0 | 50 | inference_view = single | 464 (1) | – | – | +0.060 | +0.110 | +0.046 |
| d4_training | efficientnet-b0 | 50 | inference_view = d4 | 464 (1) | – | – | +0.057 | +0.125 | +0.045 |
| d4_tta | efficientnet-b0 | 50 | train_augmentation = none | 464 (1) | – | – | +0.040 | +0.028 | +0.045 |
| d4_tta | efficientnet-b0 | 50 | train_augmentation = d4 | 464 (1) | – | – | +0.037 | +0.044 | +0.044 |
| d4_training | efficientnet-b0 | 100 | inference_view = single | 464 (1) | – | – | -0.005 | -0.065 | -0.013 |
| d4_training | efficientnet-b0 | 100 | inference_view = d4 | 464 (1) | – | – | -0.029 | -0.118 | -0.020 |
| d4_tta | efficientnet-b0 | 100 | train_augmentation = none | 464 (1) | – | – | +0.048 | +0.085 | +0.030 |
| d4_tta | efficientnet-b0 | 100 | train_augmentation = d4 | 464 (1) | – | – | +0.024 | +0.033 | +0.023 |
| d4_training | lr | 100 | inference_view = single | – (0) | 464 | – | n/a | n/a | n/a |
| d4_tta | lr | 100 | train_augmentation = none | – (0) | 464 | – | n/a | n/a | n/a |
| d4_training | scratch | 10 | inference_view = single | 464,465,466 (3) | – | – | -0.006 ± 0.011 | -0.001 ± 0.002 | -0.015 ± 0.031 |
| d4_training | scratch | 10 | inference_view = d4 | 464,465,466 (3) | – | – | -0.004 ± 0.013 | +0.006 ± 0.011 | -0.014 ± 0.029 |
| d4_tta | scratch | 10 | train_augmentation = none | 464,465,466 (3) | – | – | -0.000 ± 0.003 | -0.000 ± 0.000 | -0.000 ± 0.002 |
| d4_tta | scratch | 10 | train_augmentation = d4 | 464,465,466 (3) | – | – | +0.002 ± 0.003 | +0.007 ± 0.013 | +0.000 ± 0.001 |
| d4_training | scratch | 25 | inference_view = single | 464,465,466 (3) | – | – | -0.022 ± 0.022 | -0.004 ± 0.007 | -0.046 ± 0.043 |
| d4_training | scratch | 25 | inference_view = d4 | 464,465,466 (3) | – | – | -0.018 ± 0.023 | -0.004 ± 0.008 | -0.042 ± 0.044 |
| d4_tta | scratch | 25 | train_augmentation = none | 464,465,466 (3) | – | – | -0.004 ± 0.001 | +0.000 ± 0.000 | -0.003 ± 0.002 |
| d4_tta | scratch | 25 | train_augmentation = d4 | 464,465,466 (3) | – | – | +0.001 ± 0.001 | +0.000 ± 0.000 | +0.001 ± 0.001 |
| d4_training | scratch | 50 | inference_view = single | 464,465,466 (3) | – | – | -0.013 ± 0.025 | -0.012 ± 0.022 | -0.003 ± 0.010 |
| d4_training | scratch | 50 | inference_view = d4 | 464,465,466 (3) | – | – | -0.012 ± 0.024 | -0.004 ± 0.022 | -0.006 ± 0.007 |
| d4_tta | scratch | 50 | train_augmentation = none | 464,465,466 (3) | – | – | -0.004 ± 0.003 | -0.002 ± 0.003 | -0.002 ± 0.001 |
| d4_tta | scratch | 50 | train_augmentation = d4 | 464,465,466 (3) | – | – | -0.002 ± 0.003 | +0.006 ± 0.009 | -0.004 ± 0.004 |
| d4_training | scratch | 100 | inference_view = single | 464,465,466 (3) | – | – | +0.008 ± 0.005 | +0.001 ± 0.047 | +0.018 ± 0.023 |
| d4_training | scratch | 100 | inference_view = d4 | 464,465,466 (3) | – | – | +0.012 ± 0.005 | +0.001 ± 0.036 | +0.023 ± 0.027 |
| d4_tta | scratch | 100 | train_augmentation = none | 464,465,466 (3) | – | – | -0.004 ± 0.003 | +0.002 ± 0.010 | -0.005 ± 0.003 |
| d4_tta | scratch | 100 | train_augmentation = d4 | 464,465,466 (3) | – | – | -0.000 ± 0.003 | +0.003 ± 0.005 | -0.001 ± 0.001 |
| d4_training | svm | 100 | inference_view = single | – (0) | 464 | – | n/a | n/a | n/a |
| d4_tta | svm | 100 | train_augmentation = none | – (0) | 464 | – | n/a | n/a | n/a |
| d4_training | xgb | 100 | inference_view = single | – (0) | 464 | – | n/a | n/a | n/a |
| d4_tta | xgb | 100 | train_augmentation = none | – (0) | 464 | – | n/a | n/a | n/a |

## Individual seeds

| Model | Train % | Train augmentation | Inference | Seed | Training images | Macro-F1 | Minority macro-F1 | Accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| efficientnet-b0 | 10 | d4 | d4 | 464 | 1256 | 0.495 | 0.144 | 0.674 |
| efficientnet-b0 | 10 | d4 | single | 464 | 1256 | 0.467 | 0.156 | 0.631 |
| efficientnet-b0 | 10 | none | d4 | 464 | 1256 | 0.408 | 0.113 | 0.646 |
| efficientnet-b0 | 10 | none | single | 464 | 1256 | 0.401 | 0.088 | 0.623 |
| efficientnet-b0 | 25 | d4 | d4 | 464 | 3139 | 0.501 | 0.272 | 0.608 |
| efficientnet-b0 | 25 | d4 | single | 464 | 3139 | 0.454 | 0.220 | 0.553 |
| efficientnet-b0 | 25 | none | d4 | 464 | 3139 | 0.630 | 0.256 | 0.817 |
| efficientnet-b0 | 25 | none | single | 464 | 3139 | 0.574 | 0.199 | 0.772 |
| efficientnet-b0 | 50 | d4 | d4 | 464 | 6278 | 0.690 | 0.438 | 0.832 |
| efficientnet-b0 | 50 | d4 | single | 464 | 6278 | 0.653 | 0.394 | 0.788 |
| efficientnet-b0 | 50 | none | d4 | 464 | 6278 | 0.633 | 0.313 | 0.787 |
| efficientnet-b0 | 50 | none | single | 464 | 6278 | 0.593 | 0.284 | 0.742 |
| efficientnet-b0 | 100 | d4 | d4 | 464 | 12556 | 0.740 | 0.431 | 0.865 |
| efficientnet-b0 | 100 | d4 | single | 464 | 12556 | 0.716 | 0.399 | 0.843 |
| efficientnet-b0 | 100 | none | d4 | 464 | 12556 | 0.769 | 0.549 | 0.886 |
| efficientnet-b0 | 100 | none | single | 464 | 12556 | 0.721 | 0.464 | 0.856 |
| lr | 100 | none | single | 464 | 12556 | 0.512 | 0.145 | 0.694 |
| scratch | 10 | d4 | d4 | 464 | 1256 | 0.056 | 0.026 | 0.146 |
| scratch | 10 | d4 | d4 | 465 | 1256 | 0.030 | 0.000 | 0.109 |
| scratch | 10 | d4 | d4 | 466 | 1256 | 0.024 | 0.000 | 0.088 |
| scratch | 10 | d4 | single | 464 | 1256 | 0.051 | 0.004 | 0.147 |
| scratch | 10 | d4 | single | 465 | 1256 | 0.029 | 0.000 | 0.107 |
| scratch | 10 | d4 | single | 466 | 1256 | 0.023 | 0.000 | 0.088 |
| scratch | 10 | none | d4 | 464 | 1256 | 0.044 | 0.007 | 0.135 |
| scratch | 10 | none | d4 | 465 | 1256 | 0.042 | 0.000 | 0.118 |
| scratch | 10 | none | d4 | 466 | 1256 | 0.033 | 0.000 | 0.133 |
| scratch | 10 | none | single | 464 | 1256 | 0.044 | 0.007 | 0.135 |
| scratch | 10 | none | single | 465 | 1256 | 0.040 | 0.000 | 0.117 |
| scratch | 10 | none | single | 466 | 1256 | 0.037 | 0.000 | 0.136 |
| scratch | 25 | d4 | d4 | 464 | 3139 | 0.090 | 0.000 | 0.203 |
| scratch | 25 | d4 | d4 | 465 | 3139 | 0.136 | 0.000 | 0.274 |
| scratch | 25 | d4 | d4 | 466 | 3139 | 0.085 | 0.000 | 0.180 |
| scratch | 25 | d4 | single | 464 | 3139 | 0.088 | 0.000 | 0.203 |
| scratch | 25 | d4 | single | 465 | 3139 | 0.136 | 0.000 | 0.272 |
| scratch | 25 | d4 | single | 466 | 3139 | 0.085 | 0.000 | 0.178 |
| scratch | 25 | none | d4 | 464 | 3139 | 0.102 | 0.013 | 0.209 |
| scratch | 25 | none | d4 | 465 | 3139 | 0.135 | 0.000 | 0.303 |
| scratch | 25 | none | d4 | 466 | 3139 | 0.128 | 0.000 | 0.271 |
| scratch | 25 | none | single | 464 | 3139 | 0.107 | 0.012 | 0.214 |
| scratch | 25 | none | single | 465 | 3139 | 0.138 | 0.000 | 0.305 |
| scratch | 25 | none | single | 466 | 3139 | 0.131 | 0.000 | 0.272 |
| scratch | 50 | d4 | d4 | 464 | 6278 | 0.196 | 0.024 | 0.330 |
| scratch | 50 | d4 | d4 | 465 | 6278 | 0.204 | 0.033 | 0.356 |
| scratch | 50 | d4 | d4 | 466 | 6278 | 0.217 | 0.017 | 0.323 |
| scratch | 50 | d4 | single | 464 | 6278 | 0.195 | 0.022 | 0.330 |
| scratch | 50 | d4 | single | 465 | 6278 | 0.206 | 0.017 | 0.363 |
| scratch | 50 | d4 | single | 466 | 6278 | 0.223 | 0.016 | 0.330 |
| scratch | 50 | none | d4 | 464 | 6278 | 0.213 | 0.053 | 0.342 |
| scratch | 50 | none | d4 | 465 | 6278 | 0.236 | 0.024 | 0.362 |
| scratch | 50 | none | d4 | 466 | 6278 | 0.203 | 0.009 | 0.322 |
| scratch | 50 | none | single | 464 | 6278 | 0.221 | 0.059 | 0.345 |
| scratch | 50 | none | single | 465 | 6278 | 0.236 | 0.024 | 0.364 |
| scratch | 50 | none | single | 466 | 6278 | 0.207 | 0.009 | 0.325 |
| scratch | 100 | d4 | d4 | 464 | 12556 | 0.315 | 0.127 | 0.462 |
| scratch | 100 | d4 | d4 | 465 | 12556 | 0.342 | 0.092 | 0.506 |
| scratch | 100 | d4 | d4 | 466 | 12556 | 0.353 | 0.139 | 0.490 |
| scratch | 100 | d4 | single | 464 | 12556 | 0.318 | 0.130 | 0.464 |
| scratch | 100 | d4 | single | 465 | 12556 | 0.341 | 0.084 | 0.506 |
| scratch | 100 | d4 | single | 466 | 12556 | 0.351 | 0.136 | 0.491 |
| scratch | 100 | none | d4 | 464 | 12556 | 0.308 | 0.093 | 0.470 |
| scratch | 100 | none | d4 | 465 | 12556 | 0.325 | 0.085 | 0.474 |
| scratch | 100 | none | d4 | 466 | 12556 | 0.341 | 0.176 | 0.446 |
| scratch | 100 | none | single | 464 | 12556 | 0.310 | 0.080 | 0.472 |
| scratch | 100 | none | single | 465 | 12556 | 0.327 | 0.088 | 0.480 |
| scratch | 100 | none | single | 466 | 12556 | 0.348 | 0.179 | 0.455 |
| svm | 100 | none | single | 464 | 12556 | 0.513 | 0.149 | 0.739 |
| xgb | 100 | none | single | 464 | 12556 | 0.529 | 0.130 | 0.745 |

[Learning curves](learning_curves.pdf) · [Grouped CSV](grouped_scores.csv) · [Per-seed contrasts](paired_deltas.csv) · [Pair coverage](paired_summary.csv)
