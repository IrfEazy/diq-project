# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Academic project studying how Completeness and Distinctness data-quality issues affect linear regression (10 experiments per issue). Two notebooks drive everything: `completeness.ipynb` and `distinctness.ipynb`, backed by modules in `library/` and a LaTeX report in `Report/`.

There is no test suite, build system, requirements file, or CI. Third-party dependencies (install manually): numpy, pandas, scikit-learn, matplotlib, jupyter.

## Running

- Run notebooks from the repo root: they import `library.X` (`library/` has no `__init__.py`, it is a namespace package).
- Plots are saved to the relative path `Plot/` (`E_plot_results.your_path`). `Plot/` is not in the repo and must exist before plotting.
- Notebooks are JSON: edit with care, cell by cell with a notebook-aware tool rather than raw text edits. Headless run, as an option (not verified): `jupyter nbconvert --to notebook --execute completeness.ipynb`.

## Pipeline architecture

Modules are prefixed `A_`, `D_`, `E_`; no B or C modules exist.

1. `library/A_data_collection.py`: dataset generators wrapping sklearn `make_regression`, `make_classification`, `make_blobs`.
2. `library/my_functions.py` pollution:
   - Completeness: `pollute_data_mcar*`, `pollute_data_mar`, `pollute_data_mnar` (MCAR/MAR/MNAR missing values), plus `pollute_most/less_important_features` (feature-importance-targeted missingness, used only in `completeness.ipynb`).
   - Distinctness: `pollution_*_experiment(s)` functions (first through tenth experiment).
3. Imputation happens inside `completeness.ipynb` itself (SimpleImputer, KNNImputer, LinearRegression), not in the library.
4. `library/D_data_analysis.py`: `regression(X, y, regression, seed)` picks the model by name string (`LinearRegressor` is Ridge and also the fallback, `BayesianRidge`, `GPRegressor`, `SVMRegressor`, `KNNRegressor`, `MLPRegressor`). It uses ShuffleSplit (8 splits, test size 0.3) and returns a dict with `mean_perf`, `distance`, `speed`. It imports `ignore_warnings` from the private `sklearn.utils._testing`.
5. `my_functions.plot_results` / `zoomed_plot` (title gets suffix ' - Trial regression ' or ' - Cleaned regression ') call `library/E_plot_results.py` `plot()`, which dispatches on plot type (performance, distance train-test, speed) and saves a PDF named from the title.

Convention: `SEED = 2023`; pollution functions default to `seed=2023`, and `distinctness.ipynb` also seeds `random` and `np.random`.

## Report

`Report/Project.tex` uses the custom class `Configuration_Files/PoliMi3i_thesis.cls` and `minted`, so it needs `-shell-escape` and Pygments, e.g. `cd Report && latexmk -pdf -shell-escape Project.tex` (or `pdflatex -shell-escape`); these commands are not verified. The bibliography is an inline `thebibliography` (no `.bib`).

Figures are read from `Report/Images/{completeness,distinctness/dexN}/` (`graphicspath ./Images/`). They are referenced by filename, and no script in the repo moves them from `Plot/` into `Report/Images/`. Completeness figures are numbered `1.pdf`-`17.pdf`; distinctness figures (`dex1`-`dex10`) are named after the plot titles. The compiled `Report/DIQ_project.pdf` is tracked.

`.gitignore` ignores `**/*.pdf`, so new report figures or a rebuilt PDF must be added with `git add -f` to be tracked.
