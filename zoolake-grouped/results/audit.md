# ZooLake dataset audit

- Audit status is **ok**.
- Dataset source is the extracted Eawag ZooLake release.
- Detected the `eawag-release` layout.
- Of 17,943 image files, 17,943 decoded and 0 could not be read.
- Observed 35 labels. The ratio of largest to smallest readable class is 332.10.
- File formats {'JPEG': 17943}. Image modes {'RGB': 17943}.
- Exact matching identified 3 extra copies by file bytes and 3 by decoded RGB pixels.
- Of the decoded duplicate groups, 3 cross labels and 2 cross provider splits.
- Image width ranges from 16 to 2832 px, with a median of 72 px.
- Image height ranges from 20 to 2176 px, with a median of 76 px.
- Provider manifest row counts are {'train': 12560, 'validation': 2691, 'test': 2692}.
- The manifests repeat 3 paths, including 2 in more than one split. They leave 3 local images unassigned. The audit found 0 missing files or malformed rows.
- Official class array matches the label set and matches the local alphabetical order.

## Class support

Provider columns count singly assigned rows. The last column includes ambiguous assignments.

| Index | Label | Readable | Invalid | Unique decoded | UTC days | Provider train | Provider val | Provider test | Without one provider split |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | aphanizomenon | 225 | 0 | 225 | 12 | 164 | 33 | 28 | 0 |
| 1 | asplanchna | 607 | 0 | 607 | 110 | 410 | 103 | 93 | 1 |
| 2 | asterionella | 1055 | 0 | 1055 | 7 | 735 | 154 | 165 | 1 |
| 3 | bosmina | 80 | 0 | 80 | 38 | 51 | 15 | 14 | 0 |
| 4 | brachionus | 137 | 0 | 137 | 25 | 93 | 23 | 21 | 0 |
| 5 | ceratium | 814 | 0 | 814 | 95 | 558 | 126 | 130 | 0 |
| 6 | chaoborus | 10 | 0 | 10 | 5 | 7 | 1 | 2 | 0 |
| 7 | conochilus | 264 | 0 | 264 | 44 | 189 | 33 | 42 | 0 |
| 8 | copepod_skins | 33 | 0 | 33 | 11 | 24 | 3 | 6 | 0 |
| 9 | cyclops | 866 | 0 | 866 | 171 | 591 | 127 | 148 | 0 |
| 10 | daphnia | 721 | 0 | 721 | 148 | 510 | 99 | 112 | 0 |
| 11 | daphnia_skins | 46 | 0 | 46 | 13 | 39 | 2 | 5 | 0 |
| 12 | diaphanosoma | 1089 | 0 | 1089 | 114 | 769 | 159 | 161 | 0 |
| 13 | diatom_chain | 17 | 0 | 17 | 11 | 12 | 2 | 3 | 0 |
| 14 | dinobryon | 3321 | 0 | 3321 | 75 | 2365 | 510 | 445 | 1 |
| 15 | dirt | 131 | 0 | 131 | 17 | 91 | 18 | 22 | 0 |
| 16 | eudiaptomus | 537 | 0 | 537 | 149 | 375 | 87 | 75 | 0 |
| 17 | filament | 405 | 0 | 405 | 137 | 276 | 64 | 65 | 0 |
| 18 | fish | 222 | 0 | 222 | 32 | 155 | 29 | 38 | 0 |
| 19 | fragilaria | 306 | 0 | 306 | 29 | 215 | 56 | 35 | 0 |
| 20 | hydra | 18 | 0 | 18 | 7 | 15 | 2 | 1 | 0 |
| 21 | kellicottia | 519 | 0 | 519 | 133 | 375 | 79 | 65 | 0 |
| 22 | keratella_cochlearis | 112 | 0 | 112 | 20 | 84 | 15 | 13 | 0 |
| 23 | keratella_quadrata | 420 | 0 | 420 | 102 | 285 | 70 | 65 | 0 |
| 24 | leptodora | 203 | 0 | 203 | 28 | 144 | 26 | 33 | 0 |
| 25 | maybe_cyano | 1364 | 0 | 1364 | 2 | 958 | 202 | 204 | 0 |
| 26 | nauplius | 1507 | 0 | 1507 | 201 | 1044 | 222 | 241 | 0 |
| 27 | paradileptus | 424 | 0 | 424 | 20 | 291 | 72 | 61 | 0 |
| 28 | polyarthra | 80 | 0 | 80 | 11 | 57 | 10 | 12 | 1 |
| 29 | rotifers | 744 | 0 | 744 | 16 | 535 | 99 | 109 | 1 |
| 30 | synchaeta | 142 | 0 | 142 | 22 | 90 | 23 | 29 | 0 |
| 31 | trichocerca | 255 | 0 | 255 | 66 | 174 | 44 | 37 | 0 |
| 32 | unknown | 245 | 0 | 245 | 15 | 176 | 26 | 43 | 0 |
| 33 | unknown_plankton | 71 | 0 | 71 | 18 | 49 | 11 | 11 | 0 |
| 34 | uroglena | 953 | 0 | 953 | 13 | 652 | 145 | 156 | 0 |

## Acquisition groups

The active grouping rule is `three_minutes`.

| Group definition | Images covered | Groups | Groups with multiple images | Largest group | Cross-provider groups | Labels under three groups |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Used in split input | 17,943 | 6,162 | 2,389 | 1,055 | 1,487 | None |
| Same camera frame | 17,943 | 14,700 | 1,147 | 20 | 716 | None |
| Same UTC day | 17,943 | 362 | 330 | 1,419 | 304 | maybe_cyano (2) |
| Same UTC hour | 17,943 | 1,082 | 959 | 1,353 | 852 | maybe_cyano (2) |
| Same UTC 10-minute bin | 17,943 | 3,479 | 2,130 | 1,353 | 1,432 | maybe_cyano (2) |
| Same UTC 5-minute bin | 17,943 | 4,981 | 2,355 | 1,300 | 1,502 | None |
| Same UTC 3-minute bin | 17,943 | 6,162 | 2,389 | 1,055 | 1,487 | None |

Found 35 metadata sidecars. The JSON records their paths. Only explicit CSV groups and matching ROI filenames contribute acquisition groups.

Exact RGB hashes cover EXIF-oriented pixels at native resolution. Crop, compression, and near-duplicate variants can evade exact matching. See `audit.json` for invalid files, duplicate paths and group coverage.
