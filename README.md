# Data Information Quality Project on Completeness and Distinctness in Linear Regression

This project studies how Completeness and Distinctness data-quality issues affect regression models. Each notebook runs a series of experiments on artificial data: it pollutes the data, optionally imputes it, and evaluates six regression models with a per-fold protocol over several seeds.

## Layout

- `completeness.ipynb`: missing values (MCAR, MAR, MNAR, importance-targeted) with several imputation conditions.
- `distinctness.ipynb`: non-distinct values and duplicate records, plus the experiments on noise, binning and outliers that pollute accuracy rather than distinctness.
- `library/`: polluters, data generation, evaluation, runner, plots and a selfcheck. See `library/README.md` for the protocol and `library/HISTORY.md` for the differences from the original implementation.
- `Report/`: the LaTeX report.

## Running

```
uv sync
uv run jupyter lab                                   # run the notebooks from the repository root
uv run --frozen python -m library.selfcheck          # invariant checks
DIQ_QUICK=1 uv run --frozen jupyter nbconvert --to notebook --execute --output-dir <dir> completeness.ipynb   # smoke run
```

Results are written to `Plot/results/<exp_id>.json` and figures to `Plot/<exp_id>/` (the `DIQ_OUT_DIR` environment variable changes the directory). `Plot/` is git-ignored.
