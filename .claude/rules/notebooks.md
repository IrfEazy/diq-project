---
paths:
  - "*.ipynb"
  - "**/*.ipynb"
---

# Notebooks

- Notebooks are JSON: edit with care, cell by cell with a notebook-aware tool rather than raw text edits. Headless run, as an option (not verified): `jupyter nbconvert --to notebook --execute completeness.ipynb`.
- Experiments loop over `REGRESSION_ALGORITHMS` and `X_AXIS`.
- Imputation (SimpleImputer, KNNImputer, LinearRegression) lives in `completeness.ipynb`, not in the library.
