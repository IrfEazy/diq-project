---
paths:
  - "*.ipynb"
  - "**/*.ipynb"
---

# Notebooks

- Notebooks are JSON: edit with care, cell by cell with a notebook-aware tool (or a JSON script that keeps `indent=1` and `ensure_ascii=False`) rather than raw text edits. Headless run: `uv run --frozen jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output-dir <dir outside the repo> <notebook>.ipynb`, keeping the executed copy out of the repository, because notebooks are stored without outputs.
- `completeness.ipynb` and `distinctness.ipynb` call `library.runner.run_experiment` over `SEEDS`; imputers come from `library.evaluation` and polluters from `library.pollution`. They have a `QUICK` flag (or `DIQ_QUICK=1`) for a smoke run that truncates x axes and seeds.
- Set `DIQ_OUT_DIR` to a scratch directory for smoke runs so the stored results in `Plot/` are not overwritten. Full runs take tens of minutes each, so run them in the background and use `DIQ_QUICK=1` to check edits first.
- A change that alters results must follow the working rules in `CLAUDE.md` (justify, document in `library/HISTORY.md`, compare with the previous results).
