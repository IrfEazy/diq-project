# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Academic project studying how Completeness and Distinctness data-quality issues affect linear regression (10 experiments per issue). Two notebooks drive everything: `completeness.ipynb` and `distinctness.ipynb`, backed by modules in `library/` and a LaTeX report in `Report/`.

There is no test suite, build system, or CI. Dependencies are managed with uv (`pyproject.toml`, `uv.lock`, `.python-version`): numpy, pandas, scikit-learn, matplotlib, jupyter.

## Running

- Install dependencies with `uv sync`.
- Run notebooks from the repo root with `uv run jupyter lab` (or `uv run jupyter notebook`): they import `library.X` (`library/` has no `__init__.py`, it is a namespace package).
- Plots are saved to the relative path `Plot/` (`E_plot_results.your_path`), which is not in the repo and must exist before plotting.

## Pipeline

`library/A_data_collection` -> `my_functions` pollution -> imputation in `completeness.ipynb` -> `D_data_analysis.regression` -> `my_functions.plot_results` -> `E_plot_results`.

Convention: both notebooks set `SEED = 2023` and seed `random` and `np.random` with it.

`.gitignore` ignores `**/*.pdf`, so new report figures or a rebuilt PDF must be added with `git add -f` to be tracked.

## Scoped guidance

Loaded on demand, only when matching files are read or edited:

- `library/CLAUDE.md`: module details.
- `Report/CLAUDE.md`: LaTeX report.
- `.claude/rules/notebooks.md`: notebooks (`*.ipynb`).
