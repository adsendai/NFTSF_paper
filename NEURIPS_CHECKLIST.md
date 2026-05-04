# NeurIPS / Papers With Code ML Code Completeness Checklist

Branch: `neurips-submission-clean`  
Commit: `a4f9986`  
Date: 2026-05-04  

---

## Compliance table

| Item | Status | Location / Notes |
|------|--------|------------------|
| **1. Specification of dependencies** | ✅ | `requirements.txt` (pip, versioned); `setup_environment.sh` (conda bootstrap for HPC); Python 3.10+ and CUDA noted in `README.md § Requirements` |
| **2. Training code** | ✅ | `train_model.py` (full CLI, `--help`); `run_training.sh` (SLURM single job); `run_sweep.sh` (SLURM array sweep); `run_full_pipeline.sh` (end-to-end pipeline, all landscapes); exact paper hyperparameters documented in `README.md § Training` |
| **3. Evaluation code** | ✅ | `test_model.py` (full CLI, `--help`); `run_testing.sh` (SLURM); `run_all_tests.sh` (all landscapes); `run_compare_models.sh` (multi-model comparison); `compare.py` / `compare_models.py` (figure + CSV generation); `metrics_crps_decomp.py` (CRPS + Hersbach decomposition); `calibration_diagnostics.py`; exact eval commands in `README.md § Evaluation` |
| **4. Pre-trained models** | ⚠️ | `checkpoints/README.md` describes expected layout and usage; **download link is a TODO placeholder** — upload to Zenodo/figshare before submission |
| **5. README with results table + exact commands** | ⚠️ | `README.md` contains all required sections (Requirements, Training, Evaluation, Pre-trained Models, Results); results tables exist with correct structure and exact reproduction commands; **all metric values are TODO placeholders** — fill from paper before submission |
| **Anonymization — file content** | ✅ | Full scan performed; no emails, credentials, author names, personal URLs, or home-directory paths remain in the working tree |
| **Anonymization — commit history** | ✅ | `neurips-submission-clean` is an orphan branch with exactly **one commit** authored as `Anonymous <anonymous@example.com>`; no prior history is visible |

---

## TODOs — must be filled in by hand before submission

These items cannot be derived from the code alone and were intentionally left
as placeholders rather than fabricated.

| # | File | What to fill in |
|---|------|-----------------|
| 1 | `README.md` line 1 & 5 | Paper title (replace `NF-TSF: Normalizing Flow Time Series Forecasting` if different) and 2–3 sentence abstract snippet |
| 2 | `README.md` § Results | All CRPS / RELI / RESOL / UNCE values for each landscape × model row (copy from paper Table) |
| 3 | `README.md` line 146 | Replace `TODO: https://anonymous.4open.science/r/REPLACE_ME/` with the actual anonymous checkpoint URL |
| 4 | `README.md` § License | Add license text (MIT / Apache-2.0 recommended) |
| 5 | `checkpoints/README.md` line 10 | Replace placeholder with actual anonymous download URL after uploading to Zenodo or figshare |
| 6 | `data/README.md` line 63 | Add anonymous download link for raw `alanine-dipeptide-3x250ns-backbone-dihedrals.npz` |

---

## Publishing instructions

**Nothing has been pushed.** To publish the clean anonymous branch, run:

```bash
git push -u origin neurips-submission-clean
```

Both branches are retained locally:
- `neurips-submission` — full work history (3 commits), contains prior author metadata; do **not** push publicly
- `neurips-submission-clean` — single anonymous commit; **push this one**

---

## Items recommended for git-filter-repo (post-review, optional)

The following files appear in the **history** of `neurips-submission` (not
`neurips-submission-clean`) and would require `git filter-repo` to purge
from history entirely if that branch were ever made public:

- macOS `._*` resource-fork files (removed in commit `4a232da`)
- `unified_trajectory_forecasting/` directory (removed in commit `4a232da`)

Since `neurips-submission-clean` has no prior history, no filter-repo work
is needed for the branch you will publish.
