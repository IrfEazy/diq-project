# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Academic project studying how Completeness and Distinctness data-quality issues affect linear regression (10 experiments per issue). Three variants run side by side, each with its own notebooks and library package: v1 (`completeness.ipynb`, `distinctness.ipynb`, `library/`), v2 (`*_v2.ipynb`, `library/v2/`, faithful fixes of v1, see `library/v2/CHANGES.md`) and v3 (`*_v3.ipynb`, `library/v3/`, evaluation protocol, see `library/v3/README.md`). A LaTeX report in `Report/` presents v3.

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

## Regenerating v3 results and the report

1. Run `completeness_v3.ipynb` and `distinctness_v3.ipynb` (full runs: about 95 and 50 minutes with `N_JOBS = 8` on 8 CPUs, dominated by the Gaussian process; set `DIQ_QUICK=1` for a smoke run). They overwrite `Plot/v3/results/*.json` and the figures in `Plot/v3/<exp_id>/`.
2. `uv run --frozen python Report/tools/build_v3_assets.py` copies the selected figures and writes the tables into `Report/` (see `Report/CLAUDE.md`).
3. Build the PDF as described in `Report/CLAUDE.md`.

Scripts run from outside the repository root need the root on `sys.path` (for example `PYTHONPATH=.`), because `library` is not installed.

## Working rules

- Variants are compared, not overwritten. v1 is the original study and v2 is frozen; a change that alters results goes into a new variant (new notebooks and `library/vN/`, with a `CHANGES.md` saying what differs and why), then the results are compared. Which variant the report presents is the user's decision.
- Before fixing a pre-existing bug, classify it: unused (fix freely, behaviour is unchanged) or exploited (results depend on it, so the fix belongs in a new variant and must be documented).
- Use the latest mutually compatible library versions (`uv add`, `uv lock --upgrade`); do not pin old versions to reproduce old numbers. Explain differences instead: for example, the v1 Gaussian process results depend on the default kernel of scikit-learn <= 1.8 (see `library/v3/README.md`).
- Claims about results must be read from the results JSON or the generated tables, never from memory or summaries. Check every number before writing it in the report or docs.
- Check the theory behind an experiment (which data-quality dimension it really pollutes, which mechanism it simulates) before interpreting it. For example, the "paste an existing value" distinctness pollution is also a value-substitution (accuracy) error.

## Scoped guidance

Loaded on demand, only when matching files are read or edited:

- `library/CLAUDE.md`: module details.
- `Report/CLAUDE.md`: LaTeX report.
- `.claude/rules/notebooks.md`: notebooks (`*.ipynb`).
