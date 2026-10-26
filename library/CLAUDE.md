# library/

Namespace package (no `__init__.py`): import from the repository root as `library.<module>`. Protocol, API and results JSON are in `README.md`; what differs from the original implementation (and why) is in `HISTORY.md`.

- `pollution.py`: the data-quality polluters (completeness: MCAR/MAR/MNAR and importance-targeted missingness; distinctness and accuracy: `pollution_*_experiment(s)`). They seed the global numpy RNG (`seed=2023` default); the runner saves and restores it. Changing a polluter changes results: see the working rules in the root `CLAUDE.md`.
- `data.py`: `make_regression_data` (noisy target), duplicate-record injection and cleaning.
- `evaluation.py`: model and imputer registries, per-fold fitting, nRMSE.
- `runner.py`: `run_experiment` writes `Plot/results/<exp_id>.json` (`schema_version` 1); the output directory is `Plot` or `DIQ_OUT_DIR`, read at call time.
- `plots.py`: figures built from the results JSON only.
- `selfcheck.py`: invariant script, `uv run --frozen python -m library.selfcheck` (prints `selfcheck ok`).
