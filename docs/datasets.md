# Datasets used in R62

The code is distributed under MIT. The three evaluation datasets below are published separately under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite the data record as well as the paper when you reuse a dataset. The role and field files in the R62 companion identify the exact train, calibration and test records.

| Dataset | What it is used for | Release and downloads |
| --- | --- | --- |
| IDSpace, Slovak ID card (`SVK`) | Main few-forgery comparison, spectral and pooling ablations, the common JPEG refit, field recovery and the enlarged test set | [Pinned dataset card](https://huggingface.co/datasets/cactuslab/IDSpace/tree/a3de23b950d55a39fbd36cfaee16bd260c4d457d) · [Templates](https://huggingface.co/datasets/cactuslab/IDSpace/resolve/a3de23b950d55a39fbd36cfaee16bd260c4d457d/SVK_template.tar.gz) · [Simulated scans](https://huggingface.co/datasets/cactuslab/IDSpace/resolve/a3de23b950d55a39fbd36cfaee16bd260c4d457d/SVK_scan.tar.gz) · [Paper](https://arxiv.org/abs/2609.03052) |
| CrossVerify v1 | Registry-card label-budget comparison and both registered-layout comparisons | [Data record and download](https://doi.org/10.5281/zenodo.23136723) · [Publisher file list and checksums](https://zenodo.org/api/records/23136723) · [Pinned publisher splits and baseline code](https://github.com/kanija16/CrossVerify/tree/5f7aabc867735d4f429339519a2d97e261fb6a7f) |
| IDNet-2025, Estonia ID card (`EST`) | Supplementary native-template evaluation and encoding sensitivity experiment | [Versioned dataset card](https://huggingface.co/datasets/cactuslab/IDNet-2025/blob/f62be958aa084401f79f31f87ad11bcaf5a994bf/README.md) · [EST template archive](https://huggingface.co/datasets/cactuslab/IDNet-2025/resolve/6e6b61e8a876f2f826ff7cfecf77fe842aa94383/EST.tar.gz) · [IDNet paper](https://doi.org/10.1109/BigData62323.2024.10825017) |

## Suggested citations

- Xie, L., Wang, Y., Chowdhury, K., Garcia, R., Yang, Y., and Zou, J. (2026). *IDSPACE: A Novel Document Generator for Reliable Evaluation of Digital Identity Verification Systems [Extended Technical Report]*. [arXiv:2609.03052](https://arxiv.org/abs/2609.03052) · [IDSpace, Cactus Lab, revision `a3de23b`](https://huggingface.co/datasets/cactuslab/IDSpace/tree/a3de23b950d55a39fbd36cfaee16bd260c4d457d).
- Herbert, J., Kora, P. K., Fathima, K., R, J., and S, S. (2026). *CrossVerify: A Synthetic Identity-Document Forgery Benchmark with Visual and Cross-Modal Evidence* (v1). Zenodo. [DOI 10.5281/zenodo.23136723](https://doi.org/10.5281/zenodo.23136723).
- Cactus Lab (2025). *IDNet-2025* [data set]. [Hugging Face, revision `f62be95`](https://huggingface.co/datasets/cactuslab/IDNet-2025/blob/f62be958aa084401f79f31f87ad11bcaf5a994bf/README.md). See also Xie et al. (2024), [*IDNet: A Novel Identity Document Dataset via Few-Shot and Quality-Driven Synthetic Data Generation*](https://doi.org/10.1109/BigData62323.2024.10825017).

## File checks

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| `SVK_template.tar.gz` | 2,569,999,080 | `2ef411ce08069ed5bf859a2aa294ff951ef8c1fa908ad16d91ba52f0ba9142c7` |
| `SVK_scan.tar.gz` | 3,659,479,059 | `fe12457c293480aafbe82d743872207041fd8c0148ee3dc21ea1e24f2869a99f` |
| `EST.tar.gz` | 1,354,221,557 | `bd4c6763707fba2bb64eb8d70b65916a5dcb31c7c7109796326cf412cf142ef6` |

CrossVerify v1 is published as 46 ZIP parts. The complete joined file is 4,794,551,940 bytes, with SHA-256 `95bac9830fb4683462d20fa5a410b6532d1607eb28a00a03441c09ccff37e02f`. The runner verifies the parts against the publisher's size and checksum list and checks each selected image against the fixed R62 roster.

## Historical sources

The historical detector experiments also use these datasets. Follow the terms and access instructions in each publisher record. The registry-card refit uses the frozen historical core as an input to one comparator.

| Dataset | Source and citation |
| --- | --- |
| FantasyID | Korshunov, Mohammadi, Vidit, Ecabert, and Marcel (2025), [FantasyID paper](https://publications.idiap.ch/attachments/papers/2025/Korshunov_IJCB-2_2025.pdf) · [FantasyID data](https://zenodo.org/records/17063366) · [FantasyID-val restricted subset](https://zenodo.org/records/17063494) |
| KID34K | Park, Back, Kim, and Woo (2023), [CIKM paper](https://doi.org/10.1145/3583780.3615122) · [DASH Lab access instructions](https://github.com/DASH-Lab/idcard_fraud_detection) |
| MIDV-Holo | Koliaskina et al. (2023), [dataset description, citation and download instructions](https://github.com/SmartEngines/midv-holo) |
| SIDTD / MIDV-2020 | Boned et al. (2024), [SIDTD data descriptor](https://doi.org/10.1038/s41597-024-04160-9) · [SIDTD data](https://doi.org/10.34810/data1815) · Bulatov et al. (2021), [MIDV-2020 paper and source data locator](https://arxiv.org/abs/2107.00396) |
