# ZooLake split and subset report

Strategy **stratified**, seed **464**, grouping **three minutes**.
Assigned **17,937** readable images across **35** labels.
Quarantined **6** images in **3** decoded hashes shared by conflicting labels.
0 assigned images lacked the requested grouping field.

Exact decoded duplicates and records linked by capture frames or chosen groups share a partition.
Different frames or time blocks can still contain related crops from one capture session.

## Provider partition assessment

Provider folds eligible under the selected grouping policy **False**.
After quarantine, **0** images lacked an unambiguous provider assignment and **1487** grouped components crossed provider boundaries.

## Fixed folds

| Partition | Images | Image share | Smallest class | Grouped components |
| --- | ---: | ---: | ---: | ---: |
| train | 12,556 | 70.00% | 5 | 4,964 |
| validation | 2,690 | 15.00% | 1 | 572 |
| test | 2,691 | 15.00% | 2 | 626 |

| Class | Independent components | Train | Validation | Test |
| --- | ---: | ---: | ---: | ---: |
| aphanizomenon | 106 | 143 | 41 | 41 |
| asplanchna | 414 | 409 | 110 | 87 |
| asterionella | 13 | 868 | 48 | 138 |
| bosmina | 65 | 51 | 14 | 15 |
| brachionus | 83 | 86 | 25 | 26 |
| ceratium | 322 | 526 | 147 | 141 |
| chaoborus | 7 | 5 | 3 | 2 |
| conochilus | 96 | 177 | 42 | 45 |
| copepod_skins | 18 | 23 | 5 | 5 |
| cyclops | 599 | 563 | 155 | 148 |
| daphnia | 491 | 482 | 118 | 121 |
| daphnia_skins | 31 | 29 | 9 | 8 |
| diaphanosoma | 914 | 829 | 130 | 130 |
| diatom_chain | 14 | 10 | 4 | 3 |
| dinobryon | 1,517 | 2,257 | 595 | 467 |
| dirt | 39 | 97 | 15 | 19 |
| eudiaptomus | 415 | 346 | 97 | 94 |
| filament | 296 | 307 | 48 | 50 |
| fish | 92 | 140 | 41 | 41 |
| fragilaria | 61 | 215 | 35 | 56 |
| hydra | 9 | 14 | 1 | 3 |
| kellicottia | 300 | 332 | 94 | 93 |
| keratella_cochlearis | 63 | 79 | 12 | 21 |
| keratella_quadrata | 303 | 291 | 55 | 74 |
| leptodora | 126 | 127 | 38 | 38 |
| maybe_cyano | 3 | 975 | 164 | 225 |
| nauplius | 923 | 1,054 | 273 | 180 |
| paradileptus | 78 | 297 | 50 | 77 |
| polyarthra | 30 | 53 | 14 | 12 |
| rotifers | 142 | 566 | 88 | 89 |
| synchaeta | 49 | 102 | 16 | 24 |
| trichocerca | 231 | 187 | 38 | 30 |
| unknown | 45 | 166 | 38 | 41 |
| unknown_plankton | 33 | 47 | 14 | 10 |
| uroglena | 41 | 703 | 113 | 137 |

## Nested training subsets

Each label contributes at least one image. Remaining places are allocated by proportional deficit.
Budgets are based on the fixed training fold and expanded only if needed to cover all labels.

| Seed | Fraction | Requested | Actual | Distinct decoded images | Smallest class | Labels with expected support <1 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 464 | 10% | 1,256 | 1,256 | 1,256 | 1 | chaoborus |
| 464 | 25% | 3,139 | 3,139 | 3,139 | 1 | None |
| 464 | 50% | 6,278 | 6,278 | 6,278 | 3 | None |
| 464 | 100% | 12,556 | 12,556 | 12,556 | 5 | None |
| 465 | 10% | 1,256 | 1,256 | 1,256 | 1 | chaoborus |
| 465 | 25% | 3,139 | 3,139 | 3,139 | 1 | None |
| 465 | 50% | 6,278 | 6,278 | 6,278 | 2 | None |
| 465 | 100% | 12,556 | 12,556 | 12,556 | 5 | None |
| 466 | 10% | 1,256 | 1,256 | 1,256 | 1 | chaoborus |
| 466 | 25% | 3,139 | 3,139 | 3,139 | 1 | None |
| 466 | 50% | 6,278 | 6,278 | 6,278 | 2 | None |
| 466 | 100% | 12,556 | 12,556 | 12,556 | 5 | None |

Feasibility solver status **Optimization terminated successfully. (HiGHS Status 7: Optimal)**. Overall fold counts are within **0** images of the closest integer bounds. Per-class tolerances beyond the closest integer bounds are at least one image, or **20%** of the target rounded up.
Dominant grouped components allow a wider count tolerance for asterionella (128 images), maybe_cyano (21 images).
