# CS 464 · Sample-Efficient and Rotation-Robust Plankton Classification on ZooLake

Bilkent University · CS 464 Machine Learning · Kerem Emre Bayrak, Deniz Can Yoldaş, Mehmet Mert Suci, Osman Kaan Kasap

The TA asked us to continue with ZooLake instead of the Galaxy10 DECaLS dataset in our proposal.

| Folder | Contents |
| --- | --- |
| [`zoolake-grouped/`](zoolake-grouped/README.md) | **Main project.** Grouped split, classical models, CNNs, rotation experiments, morphology features, progress report |
| [`legacy-provider-split/`](legacy-provider-split/README.md) | First baseline on the provider split. Reference only |

## Working on the project

Start every task from `main` on its own branch, for example `kerem/imbalance`, and merge it back with a pull request. All experiments must use the shared split file once it is committed. Do not regenerate the split with the solver, because the solver can assign blocks differently on another machine.
