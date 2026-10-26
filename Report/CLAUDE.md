# Report/

`Project.tex` presents the study (setup, protocol, differences from the original study, results, conclusions, limitations). It uses the custom class `Configuration_Files/PoliMi3i_thesis.cls` and `minted`, which need `-shell-escape` and Pygments. Verified build (TeX Live 2022), run from the repository root: `latexmk -cd -pdf -shell-escape -interaction=nonstopmode -halt-on-error -outdir=<dir outside the repo> Report/Project.tex`. Even with `-outdir`, the build creates an empty `Report/_minted-Project/` and `Report/Images/*-eps-converted-to.pdf` (git-ignored, safe to delete). Without `-outdir`, it also leaves `.aux`, `.log`, `.toc`, `.fdb_latexmk` and `.fls` next to `Project.tex`. Use `\cite` (numeric `thebibliography`), not `\citet`, which would print `(author?)`. The bibliography is an inline `thebibliography` (no `.bib`).

Every number in the prose of the results chapter must come from `Tables/*.tex`.

Figures are read from `Report/Images/<exp_id>/` (`graphicspath ./Images/`, so `\includegraphics{<exp_id>/<file>.pdf}`) and the tables are `\input{Tables/<exp_id>.tex}`. Both are generated, not hand-made: after running the notebooks, run `uv run --frozen python Report/tools/build_assets.py`; it reads `Plot/results/*.json` (or `$DIQ_OUT_DIR/results`) and the selection in `Report/assets.json`. The figures and tables are tracked.

The tracked `Report/DIQ_project.pdf` is the build of `Project.tex`. It is git-ignored, so copy the rebuilt `Project.pdf` over it and add it with `git add -f`.

`build_assets.py` resolves paths from the repository root, so the working directory does not matter. It checks the manifest and that every results file and figure exists before writing anything, and ends with a clear message otherwise.

When the results change, update the prose with the new table values and re-check every number; keep claims as strong as 3 seeds allow (descriptive mean and SD, no significance claims).
