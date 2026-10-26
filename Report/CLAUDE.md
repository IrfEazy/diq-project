# Report/

`Project.tex` presents the v3 study (setup, protocol, differences from the original study, results, conclusions, limitations). It uses the custom class `Configuration_Files/PoliMi3i_thesis.cls` and `minted`, so it needs `-shell-escape` and Pygments, Verified build (TeX Live 2022 with minted/Pygments), from the repository root: `latexmk -cd -pdf -shell-escape -interaction=nonstopmode -halt-on-error -outdir=<dir outside the repo> Report/Project.tex`. Building in place (without `-outdir`) leaves `.aux`, `.log`, `.toc`, `.bbl`, `.fdb_latexmk`, `.fls` and a `_minted-Project/` directory next to `Project.tex`. The build is clean: no Overfull hbox wider than 5pt and no undefined references or citations. Use `\cite` (numeric `thebibliography`), not `\citet`, which would print `(author?)`. The bibliography is an inline `thebibliography` (no `.bib`).

Every number in the prose of the results chapter must come from `Tables/v3/*.tex`.

v3 figures are read from `Report/Images/v3/<exp_id>/` (`graphicspath ./Images/`, so `\includegraphics{v3/<exp_id>/<file>.pdf}`) and the tables are `\input{Tables/v3/<exp_id>.tex}`. Both are generated, not hand-made: after running the v3 notebooks, run `uv run --frozen python Report/tools/build_v3_assets.py` from the repository root; it reads `Plot/v3/results/*.json` and the selection in `Report/v3_assets.json`. The figures and tables are tracked.

`Images/completeness/` and `Images/distinctness/` are the v1 figures and are no longer referenced. The tracked `Report/DIQ_project.pdf` is the v3 build of `Project.tex`. It is git-ignored, so copy the rebuilt `Project.pdf` over it and add it with `git add -f`.
