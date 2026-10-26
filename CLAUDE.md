# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Academic project studying how Completeness and Distinctness data-quality issues affect regression models. `completeness.ipynb` and `distinctness.ipynb` run the experiments on top of `library/` (polluters, per-fold evaluation protocol, runner, plots; see `library/README.md`). `library/HISTORY.md` lists the differences from the original implementation, which stays available at the local git tag `pre-flatten`. A LaTeX report in `Report/` presents the results.

There is no CI and no build system. The `selfcheck` module is an invariant script, not a test runner. Dependencies are managed with uv (`pyproject.toml`, `uv.lock`, `.python-version`): numpy, pandas, scikit-learn, matplotlib, jupyter.

## Running

- Install dependencies with `uv sync`.
- Run notebooks from the repo root with `uv run jupyter lab` (or `uv run jupyter notebook`): they import `library.X` (`library/` has no `__init__.py`, it is a namespace package).
- Selfcheck: `uv run --frozen python -m library.selfcheck`.
- Outputs: results in `Plot/results/<exp_id>.json`, figures in `Plot/<exp_id>/`, created automatically. The environment variable `DIQ_OUT_DIR` overrides `Plot`. `Plot/` is git-ignored.

Notebooks take an explicit `SEEDS` list. `.gitignore` ignores `**/*.pdf`, so new report figures or a rebuilt PDF must be added with `git add -f` to be tracked.

## Regenerating results and the report

1. Run `completeness.ipynb` and `distinctness.ipynb` (full runs: about 95 and 50 minutes with `N_JOBS = 8` on 8 CPUs, dominated by the Gaussian process; set `DIQ_QUICK=1` for a smoke run, with `DIQ_OUT_DIR` pointing to a scratch directory). They overwrite `Plot/results/*.json` and the figures in `Plot/<exp_id>/`.
2. Run the asset builder in `Report/tools/` to copy the selected figures and write the tables into `Report/` (see `Report/CLAUDE.md`; it still has to be adapted to the flattened layout).
3. Build the PDF as described in `Report/CLAUDE.md`.

Scripts run from outside the repository root need the root on `sys.path` (for example `PYTHONPATH=.`), because `library` is not installed.

## Working rules

- A change that alters results must be justified, documented in `library/HISTORY.md` and verified against the previous results (a paired comparison on the same seeds). History lives in git; the original code is at the tag `pre-flatten`.
- Before fixing a pre-existing bug, classify it: unused (fix freely, behaviour is unchanged) or exploited (results depend on it, so the fix is a results-changing change under the rule above).
- Use the latest mutually compatible library versions (`uv add`, `uv lock --upgrade`); do not pin old versions to reproduce old numbers. Explain differences instead: for example, the original Gaussian process results depend on the default kernel of scikit-learn <= 1.8 (see `library/HISTORY.md`).
- Claims about results must be read from the results JSON or the generated tables, never from memory or summaries. Check every number before writing it in the report or docs.
- Check the theory behind an experiment (which data-quality dimension it really pollutes, which mechanism it simulates) before interpreting it. For example, the "paste an existing value" distinctness pollution is also a value-substitution (accuracy) error.

## Scoped guidance

Loaded on demand, only when matching files are read or edited:

- `library/CLAUDE.md`: module details.
- `Report/CLAUDE.md`: LaTeX report.
- `.claude/rules/notebooks.md`: notebooks (`*.ipynb`).
