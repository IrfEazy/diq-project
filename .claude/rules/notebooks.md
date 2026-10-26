---
paths:
  - "*.ipynb"
  - "**/*.ipynb"
---

# Notebooks

- Notebooks are JSON: edit with care, cell by cell with a notebook-aware tool rather than raw text edits. Headless run, as an option (not verified): `uv run jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir <dir outside the repo> <notebook>.ipynb`, keeping the executed copy out of the repository, because notebooks are stored without outputs.
- v1 and v2 notebooks: experiments loop over `REGRESSION_ALGORITHMS` and `X_AXIS`. Imputation (SimpleImputer, KNNImputer, LinearRegression) lives in `completeness.ipynb`, not in the library.
- v3 notebooks (`*_v3.ipynb`) call `library.v3.runner.run_experiment` over `SEEDS`; imputers come from `library.v3.evaluation`. They have a `QUICK` flag (or `DIQ_QUICK=1`) for a smoke run that truncates x axes and seeds. The v2 and v3 notebooks are stored without outputs.
- Do not overwrite or rewrite earlier-variant notebooks (v1, v2): add a new variant instead (see the working rules in `CLAUDE.md`). Full v3 runs take tens of minutes each, so run them in the background and use `DIQ_QUICK=1` to check edits first.
