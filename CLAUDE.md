# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Academic project studying how Completeness and Distinctness data-quality issues affect linear regression (10 experiments per issue). Three variants run side by side, each with its own notebooks and library package: v1 (`completeness.ipynb`, `distinctness.ipynb`, `library/`), v2 (`*_v2.ipynb`, `library/v2/`, faithful fixes of v1, see `library/v2/CHANGES.md`) and v3 (`*_v3.ipynb`, `library/v3/`, evaluation protocol, see `library/v3/README.md`). A LaTeX report lives in `Report/`.

There is no CI and no build system. The `selfcheck` modules are invariant scripts, not a test runner. Dependencies are managed with uv (`pyproject.toml`, `uv.lock`, `.python-version`): numpy, pandas, scikit-learn, matplotlib, jupyter.

## Running

- Install dependencies with `uv sync`.
- Run notebooks from the repo root with `uv run jupyter lab` (or `uv run jupyter notebook`): they import `library.X` (`library/` has no `__init__.py`, it is a namespace package).
- Selfchecks: `uv run --frozen python -m library.v2.selfcheck` and `uv run --frozen python -m library.v3.selfcheck`.
- Plots: v1 saves to the relative path `Plot/` (`E_plot_results.your_path`), which must exist before plotting. v2 writes `Plot/v2/` and v3 writes `Plot/v3/` (with `results/*.json`), both created automatically. `Plot/` is git-ignored.

## Pipeline (v1)

`library/A_data_collection` -> `my_functions` pollution -> imputation in `completeness.ipynb` -> `D_data_analysis.regression` -> `my_functions.plot_results` -> `E_plot_results`.

Convention: the v1 and v2 notebooks set `SEED = 2023` and seed `random` and `np.random` with it. v3 notebooks take an explicit `SEEDS` list.

`.gitignore` ignores `**/*.pdf`, so new report figures or a rebuilt PDF must be added with `git add -f` to be tracked.

## Scoped guidance

Loaded on demand, only when matching files are read or edited:

- `library/CLAUDE.md`: module details.
- `Report/CLAUDE.md`: LaTeX report.
- `.claude/rules/notebooks.md`: notebooks (`*.ipynb`).
