# Section 3.7 field-classifier verification

The extension refits the document branch using field-specific genuine residuals, with the original joint detector's alignment, offset, strength unit, and fallback mean. Its branch score scales come from genuine training rows. A separate genuine calibration set supplies the final threshold. The two variants require separate calibration even when they use the same training and calibration rosters.

## Archived context checks

The package refit the models in memory from the archived training features, preassigned folds, and training tags. It scored the archived calibration and test features in their recorded order. The original role assignments and recovery masks were reused. CrossVerify used its fixed field boxes. IDSpace used the frozen original offset and archived registered native rectangles. Each fitted score vector was compared with the existing research runner's output before any summary was computed.

| Layout and context | Genuine CAL | TEST | Max CAL score error | Max TEST score error | Matched strictly above-cut decisions |
| --- | ---: | ---: | ---: | ---: | --- |
| CrossVerify registry-card `b15_d0` | 75 | 284 | 0 | 0 | 5%, 10% |
| CrossVerify registry-card `b120_d2` | 75 | 284 | 0 | 0 | 5%, 10% |
| IDSpace SVK `f0_d3_o0` | 500 | 7,126 | 1.39e-13 | 6.82e-12 | 1%, 5%, 10% |
| IDSpace SVK `f1_d3_o0` | 500 | 7,206 | 2.71e-13 | 2.18e-11 | 1%, 5%, 10% |

CrossVerify original and refitted document coefficients, genuine document score scales, and full-extension held score order matched exactly. The native pixel measurements also matched exactly for all 1,077 ordered field crops over 75 calibration and 284 test image rows. IDSpace refitted document coefficients differed by at most 6.35e-11, and the genuine document score references by at most 1.75e-10. Fallback means and strength units matched exactly. IDSpace TEST includes the original and expanded populations. The `f1_d3_o0` packet includes one image with unavailable local fields, for which the strict decisions also matched.

The comparison used NumPy `assert_allclose` with absolute tolerance `1e-9`, relative tolerance `1e-10` for CrossVerify scores and `1e-9` for IDSpace scores. Decisions were compared exactly using separately recomputed calibration cuts at ascending ranks 75 and 73 (CrossVerify) and 499, 484 and 462 (IDSpace). The run used Python 3.14.4, NumPy 2.4.4, Pillow 12.2.0, and OpenCV 5.0.0, with two BLAS threads on Windows. This covers four cached-feature contexts and the observed native crop check. Full raw-image reproduction also requires the frozen encoder and geometry pipelines described in the research guides.

## Inputs and reproduction

Run `python -m pytest tests/test_field_classifiers.py -q` from the installed package's source checkout for the synthetic checks, including the native measurement test when OpenCV is installed. For archived input provenance and the study-specific fitting commands, see `field_transfer_evidence` in the R62 submission archive and the [reproduction guide](reproduction.md). The workspace runners were `research/nrsw_generalization_20261114/crossverify_budget/run_budget.py` and `research/nrsw_generalization_20261114/idspace/run.py`. They describe the complete requirements for locating licensed imagery, archived descriptors, roster and geometry files. For the package comparison, adapt their per-context training packet to `nrsw.kernel.TrainBatch`, call `kernel.fit_joint` with the archived historical mean (and `frozen_offset` for IDSpace), and call `field_classifiers.fit_field_classifiers` with that model and matching pixel features and availability. Compare `FieldClassifierModel.score` with the registry-card `R55_full` row or IDSpace's third score row, retaining the packet order and the variant's own thresholds.

| Source artifact | SHA-256 |
| --- | --- |
| CrossVerify budget `run_budget.py` | `03ef89a14f0321068dfbaf05d4e770bd33467c5e257edf0428e30140ac738f92` |
| IDSpace transfer `run.py` | `c980cf1674aba3f648f4e46df5a6603372c661e809e3aec67106081379a24ecd` |
| Archived native pixel operator `pixel_fields.py` | `317f5fec329cf5e86c39188c794aff8c957c3cd776c590a9eaf54ecf3d942f13` |
| CrossVerify budget `run_001/FIT_COMPLETE.json` | `69005095b1c53b95f6ad3437f8d561d3c7d31af98b21cb6d87d221ec8f086579` |
| CrossVerify budget `run_001/CAL_COMPLETE.json` | `e8bbd62620e76d864b9bea7ff7101f2210620974530a83cdcc49b9e294736af9` |
| CrossVerify budget `run_001/TEST_COMPLETE.json` | `40f4d87d4cc1105b5aac1d15909dc228254c09d23f3f1e371cdd854a2cc02cb8` |
| IDSpace transfer `idspace_r55_001/FIT_COMPLETE.json` | `9e91f8f68db3b594a397c4d2279d83a83a4b60e28e752627fbde091c0539c365` |
| IDSpace transfer `idspace_r55_001/CAL_COMPLETE.json` | `e6ef1fa65ee23f2c0801f82501ef4d38fd0b1f9382c63b47b23ea6adc88ac0df` |
| IDSpace transfer `idspace_r55_001/TEST_COMPLETE.json` | `f3f75300ab10768e8187114cf416c7c689e62a2afde2d517892203dbd48714d0` |
