---
paths:
  - "*.ipynb"
  - "**/*.ipynb"
---

# Notebooks

- Notebooks are JSON: edit with care, cell by cell with a notebook-aware tool rather than raw text edits. Headless run, as an option (not verified): `jupyter nbconvert --to notebook --execute completeness.ipynb`.
- v1 and v2 notebooks: experiments loop over `REGRESSION_ALGORITHMS` and `X_AXIS`. Imputation (SimpleImputer, KNNImputer, LinearRegression) lives in `completeness.ipynb`, not in the library.
- v3 notebooks (`*_v3.ipynb`) call `library.v3.runner.run_experiment` over `SEEDS`; imputers come from `library.v3.evaluation`. They have a `QUICK` flag (or `DIQ_QUICK=1`) for a smoke run that truncates x axes and seeds. The v2 and v3 notebooks are stored without outputs.
