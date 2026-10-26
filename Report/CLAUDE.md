# Report/

`Project.tex` presents the v3 study (setup, protocol, differences from the original study, results, conclusions, limitations). It uses the custom class `Configuration_Files/PoliMi3i_thesis.cls` and `minted`, so it needs `-shell-escape` and Pygments, e.g. `cd Report && latexmk -pdf -shell-escape Project.tex` (or `pdflatex -shell-escape`); these commands are not verified, and the v3 body has only been checked statically (balanced environments and braces, existing `\input`/`\includegraphics` paths, defined refs and cite keys), not compiled. The bibliography is an inline `thebibliography` (no `.bib`).

Every number in the prose of the results chapter must come from `Tables/v3/*.tex`.

v3 figures are read from `Report/Images/v3/<exp_id>/` (`graphicspath ./Images/`, so `\includegraphics{v3/<exp_id>/<file>.pdf}`) and the tables are `\input{Tables/v3/<exp_id>.tex}`. Both are generated, not hand-made: after running the v3 notebooks, run `uv run --frozen python Report/tools/build_v3_assets.py` from the repository root; it reads `Plot/v3/results/*.json` and the selection in `Report/v3_assets.json`. The figures and tables are tracked.

`Images/completeness/` and `Images/distinctness/` are the v1 figures and are no longer referenced. The tracked `Report/DIQ_project.pdf` is the old v1 build until the report is recompiled.
