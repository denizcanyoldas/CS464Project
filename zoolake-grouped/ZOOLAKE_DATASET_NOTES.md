# ZooLake data provenance

The handoff’s original notes were absent. This file records the independently verified sources and the archive used for the progress run.

| Source | Use |
| --- | --- |
| [Kyathanahally et al. (2021), *Deep Learning Classification of Lake Zooplankton*](https://doi.org/10.3389/fmicb.2021.746297) | Primary dataset paper. It documents 17,943 regions of interest, 35 classes, the lower-magnification Dual Scripps Plankton Camera at Lake Greifensee, and the rationale for aspect-preserving black padding. |
| [Eawag Open Data dataset record](https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7) and [original archive](https://opendata.eawag.ch/dataset/52b6ba86-5ecb-448c-8c01-eec7cb209dc7/resource/1cc785fa-36c2-447d-bb11-92ce1d1f3f2d/download/data.zip) | The archive used for experiments. Its SHA-256 is `ce038e19af73dd5aff0beb9ed0dc012bb0314efeb341337667b62d1e1c383d4f`. |
| [Eyring et al. (2025), *Five years of high-frequency data of phytoplankton zooplankton and limnology from a temperate eutrophic lake*](https://doi.org/10.1038/s41597-025-04988-9), Figure 3b, and [SPCConvert source](https://github.com/tooploox/SPCConvert/blob/master/spcconvert.py) | Support interpreting the timestamp embedded in original ROI filenames. UTC bins are a decision in this project. |
| [Original Plankifier code and archive structure notes](https://github.com/mbaityje/plankifier) | Documents the per-class `training_data` folders and accompanying feature sidecars. |
| [OcéanIA ZooLake dataset card](https://huggingface.co/datasets/project-oceania/zoolake) | Lists the original CC BY 4.0 license and derivative split supports. The hosted derivative is gated and was unnecessary for this run. |

The archive has an outer `data/` directory. JPEGs live under `data/zooplankton_0p5x/<label>/training_data/`. The provider manifests and Unicode label array live in the sibling `data/zoolake_train_test_val_separated/` directory. The provider lists have stale absolute paths. The adapter matches exact release-relative paths and reads the NumPy array with pickle disabled. The feature sidecars, summary figures and serialized `Data.pickle` are outside the model inputs.

The [audit summary](results/audit.md) records observed labels, image geometry, and the evidence for frame and time-block grouping. Run `python -m zoolake audit` to generate `artifacts/audit/audit.json` with the full inventory and original partition memberships.
