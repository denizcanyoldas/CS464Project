# EfficientNet-B0 validation replication

Twelve fresh fits crossed two training budgets, two training augmentation settings and seeds 464–466. Each augmentation pair used the same training images and seed. The 25% budget used 3,139 images and the 100% budget used 12,556 images. Every run used the same 2,690 validation images across 35 classes. The shared group map reconstructed the original split assignments and nested subsets without running the solver. [Split identity](split_identity.json)

ImageNet-initialized EfficientNet-B0 used 96-pixel aspect-preserving inputs, batches of 64, AdamW at 0.001, class-balanced cross-entropy and six epochs. Single-view validation macro-F1 selected the checkpoint. Each checkpoint was scored with one view and with probabilities averaged across eight D4 views. Training and validation inference took 26.48 minutes in total on Apple Metal. [Study config](../../configs/efficientnet_replication.toml)

## Validation scores

Macro-F1 is the mean ± sample standard deviation across three seeds.

| Training budget | Training augmentation | Single view | D4 TTA |
| --- | --- | ---: | ---: |
| 25% | None | 0.5466 ± 0.0256 | 0.5830 ± 0.0423 |
| 25% | D4 | 0.4544 ± 0.0817 | 0.4946 ± 0.0923 |
| 100% | None | 0.7100 ± 0.0161 | 0.7527 ± 0.0143 |
| 100% | D4 | 0.7251 ± 0.0137 | 0.7595 ± 0.0195 |

- With inference held at one view, D4 training changed macro-F1 by **−0.0922** on average at 25% (1/3 seeds improved) and **+0.0150** at 100% (2/3 seeds improved).
- D4 TTA improved macro-F1 in **all 12** matched checkpoint comparisons. The gains averaged +0.0345 to +0.0427 across the two full-budget conditions. Over 2,690 validation images, average model forward and probability return took 1.28 s for one view and 10.16 s for eight views.
- At 100% with TTA, the nine minority labels averaged 0.5192 macro-F1 without D4 training and 0.5181 with D4 training. The 0.0068 overall macro-F1 difference between those full-budget TTA conditions changed sign across seeds.

Seed standard deviations describe variation on this fixed split. The replication uses validation scores throughout. The original study’s locked test result remains in the [original study](../final-study/README.md).

[Full score tables](validation/summary.md) · [Learning curves](validation/learning_curves.pdf) · [Per-seed D4 differences](validation/paired_deltas.csv). The `runs/` directories retain settings, model hashes, epoch logs and per-class validation metrics for all 12 fits.
