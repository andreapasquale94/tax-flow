#!/usr/bin/env bash
# Rebuild reachability_report.pdf from the markdown source.
# Requires: pandoc >= 3, xelatex (texlive-xetex), TeX Gyre fonts (texlive-fonts-recommended)
set -e
cd "$(dirname "$0")"
pandoc reachability_report.md \
  --pdf-engine=xelatex \
  --toc \
  --toc-depth=2 \
  --number-sections \
  -V toc-title="Contents" \
  -o reachability_report.pdf
echo "Done: reachability_report.pdf"
