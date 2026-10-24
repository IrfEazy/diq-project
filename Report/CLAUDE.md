# Report/

`Project.tex` uses the custom class `Configuration_Files/PoliMi3i_thesis.cls` and `minted`, so it needs `-shell-escape` and Pygments, e.g. `cd Report && latexmk -pdf -shell-escape Project.tex` (or `pdflatex -shell-escape`); these commands are not verified. The bibliography is an inline `thebibliography` (no `.bib`).

Figures are read from `Report/Images/{completeness,distinctness/dexN}/` (`graphicspath ./Images/`). They are referenced by filename, and no script in the repo moves them from `Plot/` into `Report/Images/`. Completeness figures are numbered `1.pdf`-`17.pdf`; distinctness figures (`dex1`-`dex10`) are named after the plot titles. The compiled `Report/DIQ_project.pdf` is tracked.
