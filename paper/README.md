# Paper: Set-Based Uncertainty Propagation in Orbital Dynamics

A self-contained methods note on combining Automatic Domain Splitting (ADS) with
zonotope and polynomial-zonotope set representations for orbital-uncertainty
propagation, grounded in the two-body examples of this repository.

- `main.tex` — the manuscript source.
- `references.bib` — bibliography (all entries cross-checked against their
  published venue/DOI).
- `figures/` — committed figure PDFs (so the paper builds without regenerating
  data) and `make_figures.py`, which renders every figure from the experiment
  JSON using only the Matplotlib **Blues** colormap.
- `main.pdf` — the compiled manuscript (10 pages).

## Build the PDF

```bash
latexmk -pdf main.tex     # or: pdflatex main && bibtex main && pdflatex main x2
```

## Regenerate the figures

The figures are driven by JSON produced by the two-body example programs. From
the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTAXFLOW_BUILD_EXAMPLES=ON
cmake --build build -j

mkdir -p paper/data && cd paper/data
for ex in zonotope_representations two_body_ads two_body_zonotope \
          two_body_zonotope_adaptive two_body_poly_zonotope zonotope_two_body_mc; do
  ../../build/examples/$ex
done
cd ../figures && python3 make_figures.py     # writes fig_*.pdf
```

The `paper/data/` directory is git-ignored (≈6 MB, regenerable); the figure PDFs
are committed so the manuscript compiles out of the box.
