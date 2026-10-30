# Reproducing the SIST 2026 results

Requires ADNI access. All commands assume the repository root as the working
directory and a `work/` folder that is ignored by git.

| Step | Command |
|---|---|
| Cohort | `adfusion cohort mri_export.csv pet_export.csv --out work/cohort` |
| Preprocessing | `adfusion preprocess --raw /data/ADNI --labels mri_export.csv pet_export.csv --out work/cache_reg` |
| Whole-brain run | `adfusion train --manifest work/cache_reg/cache_manifest.csv --out work/results_full --config configs/protocol_sist2026.json` |
| ROI cache | `adfusion roi-cache --manifest work/cache_reg/cache_manifest.csv --out work/cache_roi` |
| ROI run | `adfusion train --manifest work/cache_roi/cache_manifest.csv --out work/results_roi --config configs/protocol_sist2026.json` |
| Modality-specific inputs | `adfusion mixed --full work/cache_reg/cache_manifest.csv --roi work/cache_roi/cache_manifest.csv --out work/cache_mixed` |
| Main run | `adfusion train --manifest work/cache_mixed/cache_manifest.csv --out work/results_mixed --config configs/protocol_sist2026.json` |
| Figures | `adfusion figures work/results_mixed` |

Expected cohort: 168 subjects (83 AD, 85 CN). Expected registration QC: mean
Dice about 0.73. Exact AUC values can differ slightly between GPU models and
library versions because cuDNN convolutions are not bit-reproducible; the
protocol file, `run_config.json` and the recorded library versions make such
differences traceable.
