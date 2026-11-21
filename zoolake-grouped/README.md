# Sample-Efficient and Rotation-Robust Plankton Image Classification on ZooLake

Bilkent University · CS 464 Machine Learning

The team reports that the TA requested ZooLake after submission of the [Galaxy10 DECaLS proposal](references/SUBMITTED_GALAXY10_PROPOSAL.pdf). ZooLake is the current implementation target. Whether the final deliverable should include Galaxy10 as a second dataset remains open. See [research decisions and TA questions](DECISIONS.md).

## Progress milestone

| Measured item | Result |
| --- | ---: |
| Original Eawag images and labels | 17,943 / 35 |
| Unreadable files | 0 |
| Quarantined images with contradictory duplicate labels | 6 across 3 pairs |
| Fixed train / validation / final test images | 12,556 / 2,690 / 2,691 |
| Weighted LR validation macro-F1 | **0.5122** |
| Minority validation macro-F1 | 0.1445 |
| Validation accuracy | 69.37% |
| Full-feature extraction, fit and validation prediction on CPU | 10.8 s |

The result uses one **full-training, no-augmentation, single-view** logistic regression run with seed 464. The progress report uses the validation fold. Final-test scores are reserved until model and inference settings are frozen.

- [Progress report PDF](deliverable/progress_report.pdf) and [editable LaTeX source](report/progress_report.tex)
- [Audit summary](results/audit.md) and [split summary](results/split_summary.md)
- [Validation metrics](results/metrics.json), [F1 by class with support](results/per_class_f1.png), and [run provenance](results/run.json)

The audit inventory, fixed split manifest, trained model, and predictions are generated locally under `artifacts/` by the commands below.

## Reproduce the baseline

Run from the repository root. The raw data and Python environment are in sibling directories `../data` and `../.venv`. The project was exercised with Python 3.13.3.

Activate the existing environment with `source ../.venv/bin/activate`. For a fresh setup, run the following.

```bash
python3 -m venv ../.venv
source ../.venv/bin/activate
python -m pip install -r requirements.txt
python -m zoolake hardware
```

The original archive is already present in this workspace. On a clean copy, download the [Eawag release](https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7) into `../data/eawag-zoolake.zip` and extract it as follows. Allow about 4 GB for the archive and extracted release together. [Dataset notes](ZOOLAKE_DATASET_NOTES.md) provide the direct download and SHA-256.

```bash
mkdir -p ../data/zoolake
python -m zipfile -e ../data/eawag-zoolake.zip ../data/zoolake
python -m zoolake audit
python -m zoolake split
MPLBACKEND=Agg python -m zoolake train --out artifacts/runs/reproduction-lr-s464
python -m unittest discover -s tests -v
```

To use another extraction location, set `ZOOLAKE_DATA_ROOT` or pass `--data-root /path/to/extracted/archive` to `audit` and `train`. Use the same relative directory level in both commands. Missing or unreadable paths produce a diagnostic and audit files with the failure status. The official structure resolves automatically. The adapter also supports label folders and `train|validation|test/<label>/...` folders. For a custom layout or better verified capture metadata, pass `audit --metadata metadata.csv` with `path,label` and optional `split,group` columns. Give one image path relative to the selected data root per row. An explicit group takes precedence over timestamp groups.

### Split and preprocessing

The provider lists repeat two paths across boundaries. Also, 716 camera-frame groups and 1,487 three-minute groups cross their partitions. A day-, hour-, or ten-minute-disjoint split could give `maybe_cyano` to at most two folds. Five-minute groups leave only 11 examples of that class in their smallest block. Three-minute UTC blocks give it 975, 225 and 164 images and make all-label coverage feasible. The grouped integer split uses seed 464, preserves exact decoded duplicates and original frames within folds, and quarantines both copies of each conflicting pair. Same-session relationships across blocks and near duplicates remain possible. The audit and split summaries retain the diagnostic counts and allocation tolerances.

```bash
# Reproduce the provider rejection and day-level infeasibility
python -m zoolake split --strategy provider --out artifacts/provider-check
python -m zoolake split --group-by acquisition_day --out artifacts/day-check
```

The fixed fold has all 35 labels in each partition. Seeds 464–466 yield nested subsets of **1,256 / 3,139 / 6,278 / 12,556** training images. For each seed, every class gets one place before filling the budget by proportional deficit. Proportional 10% allocation gives `chaoborus` 0.5 expected images. Its single assigned training example keeps the label available while leaving the overall budget intact.

All models shrink oversized ROIs without changing their aspect ratio and center them on a black square. The LR model fits a scaler and an L2-regularized multinomial classifier on 104 handcrafted features from training only. Validation output uses all 35 labels to compute macro-F1. The minority cohort is the bottom quarter by full-training support, including ties. It contains nine labels in this split. Models, hashes and configuration are saved in each run directory.

## Staged experiments

1. **Time two pilots.** Run one small scratch CNN trial and one EfficientNet-B0 trial with a single epoch. This Mac has an M2 Pro with 10 CPU cores, a 16-core Metal GPU, and 32 GB of unified memory. PyTorch 2.10.0 can allocate an MPS tensor in a native process. The restricted sandbox gives CPU only and there is no CUDA. At 100%, one epoch is 197 train batches at batch size 64 and 43 one-view validation batches. In a four-thread synthetic-input benchmark at 128 px and batch 64, train/validation batches took 0.30/0.13 s for scratch and 16.48/1.35 s for EfficientNet-B0. That projects to roughly **1 minute or 55 minutes per full-budget CPU epoch**, respectively, before image loading and eight-view scoring. Time a real native MPS or CUDA pilot before scheduling the EfficientNet grid.
2. **Establish a paired comparison.** Start with scratch at the 25% and 100% fractions and seed 464. Add the EfficientNet pair once pilot timing fits the budget. Match each pair's rows, initialization seed, preprocessing, weighting, learning rate, and epochs. Train with no augmentation and with a uniformly sampled D4 transform per CNN example and epoch. Compare one-view inference to eight-view probability averaging on each selected checkpoint. Record per-class F1 and support. Use paired differences and report performance by orientation if making a claim about robustness beyond D4. Eight-view scoring processes 21,520 views per validation run.
3. **Extend where feasible.** Add 10% and 50% and repeat the principal paired comparison for seeds 465 and 466 as compute allows. Align the seed IDs in the curve CSV for paired comparisons. Run RBF-SVM and XGBoost on the same features if resource estimates allow. Use single-copy training for the SVM baseline. Its internal probability calibration uses row-level folds, so D4 copies of a source can be shared between its internal fitting and calibration folds. Training with D4 copies also increases fit cost substantially.
4. **Freeze and evaluate.** Choose models and inference views using validation and record the choice before opening the holdout. `evaluate-test` reloads the frozen model, verifies model and split hashes, and scores test without fitting. Use `--view single` or `--view d4` for one selected inference mode. Use `--view both` only for a planned comparison. Report the chosen protocol with the scores.

```bash
python -m pip install -r requirements-cnn.txt
python -m zoolake train --model scratch --fraction 25 --epochs 1 --out artifacts/runs/pilot-scratch
python -m zoolake train --model efficientnet-b0 --fraction 10 --epochs 1 --out artifacts/runs/pilot-efficientnet

# Example of the matched pair after timing. The command defaults are in configs/default.toml.
python -m zoolake train --model scratch --fraction 25 --seed 464 --augment none --tta --out artifacts/runs/scratch-f25-none
python -m zoolake train --model scratch --fraction 25 --seed 464 --augment d4 --tta --out artifacts/runs/scratch-f25-d4

python -m zoolake train --model svm --fraction 10 --out artifacts/runs/svm-f10
python -m pip install -r requirements-xgb.txt
python -m zoolake train --model xgb --fraction 25 --out artifacts/runs/xgb-f25

# Once the paired runs exist, plot macro-F1 by budget. Error bars show seed SD when repeated.
python -m zoolake plot-curve --runs artifacts/runs/scratch-f25-{none,d4} --out artifacts/curves/scratch

# After selecting a frozen run
python -m zoolake evaluate-test --run artifacts/runs/SELECTED_RUN --view single --out artifacts/final-test
```

EfficientNet-B0 downloads its ImageNet weights on the first run unless they are cached. The CPU baseline is sufficient for the initial progress milestone. The experiment grid is unrun.

## Report source

The progress report is dated 20 November and due **Sunday, 29 November 2026 at 23:59**. It includes the four names and student numbers from the proposal and gives each member a workstream. Mert’s audit and split verification steps are in [the research decisions](DECISIONS.md#presentation-assignment-for-mehmet-mert-suci).

```bash
python report/make_figures.py
cd report
tectonic --outdir ../deliverable progress_report.tex
```

The figure script defaults to the reported run under `artifacts/runs/baseline-lr-s464`. To use the reproduction, pass `--metrics artifacts/runs/reproduction-lr-s464/metrics.json`. The report source and its figures are under `report/`.
