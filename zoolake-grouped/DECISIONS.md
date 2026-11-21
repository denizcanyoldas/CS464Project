# Research decisions

**Bilkent CS 464 · 20 November 2026**

## Confirmed for this implementation

| Item | Decision and evidence |
| --- | --- |
| Submitted proposal | The [submitted one-page proposal](references/SUBMITTED_GALAXY10_PROPOSAL.pdf) confirms the original Galaxy10 DECaLS target. It also documents the nested fractions, matched D4 experiments, proposed model families, and evaluation metrics that inform the ZooLake study. The progress report uses the four names and student numbers from this PDF. |
| TA feedback | The team reports that the TA requested ZooLake. The original message was unavailable. The working implementation and progress report use ZooLake. |
| Dataset | The original Eawag archive was downloaded and audited. Its SHA-256, citation and layout are in [ZOOLAKE_DATASET_NOTES.md](ZOOLAKE_DATASET_NOTES.md). |
| Research question | How does macro-F1 change as the labeled training budget shrinks, and what do D4 training augmentation and eight-view inference contribute at a fixed budget? |
| Split | Use the new seed-464 grouped 70/15/15 split. Six files with conflicting duplicate labels are quarantined. All 35 labels appear in all folds. Three-minute UTC windows and same-frame IDs are confined to one fold. Whole-session independence remains unverified. |
| Provider folds | Rejected for this protocol after auditing repeated paths and cross-fold capture groups. The audit retains provider memberships for traceability. |
| Training fractions | Use nested 10/25/50/100% image budgets within each of seeds 464–466 and one fixed split across all runs. The 10% budget would allocate 0.5 samples proportionally to `chaoborus`. Assign one per class first and fill the remaining budget by proportional deficit. Report actual support. |
| Evaluation | Macro-F1 across all 35 labels is primary. Report accuracy, per-class support/F1 and minority macro-F1. The minority cohort is determined by the lowest quarter of full-training support, including ties. Train and choose settings with validation evidence. Evaluate the frozen configuration on the final test after selection. |
| Progress report | Due Sunday, 29 November 2026 at 23:59. The current result is one real weighted logistic regression run on validation. The workload section assigns each member a verification or follow-up workstream. |

## Open with the TA

1. Does ZooLake fully replace Galaxy10 DECaLS, or should the final report compare two datasets?
2. If a benchmark split is required, which release and grouping protocol should be used given the original provider lists’ overlap? Our progress result is explicitly tied to the three-minute grouped split.
3. Is the smallest-budget experiment acceptable with a one-image minimum per class, and should a separate stronger temporal evaluation be included if it can retain only a restricted, explicitly reported set of labels?

## Conditional experiments

- Run matched no-augmentation and D4 training CNNs on the same rows for each seed/fraction pair. Use eight-view mean probabilities as a separate inference comparison on the same checkpoint. A scratch CNN and ImageNet-pretrained EfficientNet-B0 support the comparison.
- Time one pilot per architecture before scheduling additional fractions or seeds. Repeat the principal comparison on three seeds if practical. Report observed seed variation and raw per-class supports.
- Run LR, RBF-SVM and XGBoost on the same handcrafted representation as time allows. Treat full-size probability-calibrated RBF-SVM and classical eight-copy D4 training as optional after timing.

## Final study protocol

- The original handoff archive was located after the initial implementation. It confirms that the planned stages are a staged study with compute-dependent repetitions. ZooLake remains the working dataset under the reported TA feedback. The submitted proposal remains unchanged.
- Preserve the existing three-minute grouped split and exact split manifest for all study models. Keep the two project implementations independent. The provider-fold model from the team’s other implementation has training examples in this study’s final test and is outside the candidate set.
- Freeze the run matrix in `configs/final_study.toml`. Scratch CNN uses all four nested fractions, paired none/D4 training and seeds 464–466. EfficientNet-B0 uses the same fractions and augmentation arms with seed 464. Both CNNs use 96-pixel aspect-preserving inputs and six epochs on native Metal. The resolution reflects the pilot timing and is shared by CNN arms. Checkpoints are selected on single-view validation macro-F1.
- Compare single-view inference to mean probabilities across eight D4 views on the same frozen checkpoint. Forward timings start after the decoded batch is on the device. They exclude data loading and include the synchronization needed to return probabilities.
- The three classical models use the 104-feature 128-pixel pipeline described in the progress report. The same feature code supports weighted LR, RBF-SVM and XGBoost. The learning curves focus on the CNNs. The imbalance comparison uses scratch CNN with D4 at 10% and 100%, seed 464, and switches loss weights while preserving all other settings.
- The angle sweep scores 0° through 315° in 45° increments. Every angle shares one guarded canvas, including angle zero. This keeps the input field of view consistent within the sweep. Canonical validation uses the original 96- or 128-pixel preprocessing and has a separate result.
- Choose the final run from the explicit list of completed full-budget, seed-464 validation runs by highest macro-F1 across saved inference views. Use single-view on exact ties and stable run name thereafter. Freeze the run, model, validation and split hashes, checkpoint device, and selected inference view before evaluating test. Compare all other configurations on validation.
- Long sessions may still cross blocks. The [near-duplicate screen](results/near_duplicate_screen.json) adds an aligned train/validation sensitivity diagnostic while preserving the split. Its candidates are visually similar images with unknown capture identity.

## Presentation assignment for Mehmet Mert Suci

Use the audit summary to check the six files with conflicting duplicate labels. Then regenerate the split from the saved audit and compare it to the report’s manifest. Run this from the project folder with the environment activated. Use a fresh output path for another check.

```bash
python -m zoolake split --out artifacts/mert-split-check
cmp artifacts/splits/splits.json artifacts/mert-split-check/splits.json
```

Use the three questions below to walk through the decisions in the presentation.

- **Why replace the provider folds?** Two paths repeat across folds, and after quarantine 1,487 three-minute components still cross the provider boundaries.
- **Why three minutes?** The `maybe_cyano` label is present in only two days, hours and ten-minute bins. Three-minute grouping preserves all 35 labels in each fold and keeps original frames together. It still permits a session to span folds.
- **How does the 10% subset retain every label?** The rarest training class has five images. A tenth of five yields 0.5, so the allocation reserves one per class and fills the remaining places under the original 1,256-image budget.
