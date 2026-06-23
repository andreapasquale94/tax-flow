#!/usr/bin/env bash
# Rebuild combined_report.pdf from the markdown source.
# Requires: pandoc >= 3, xelatex (texlive-xetex), TeX Gyre fonts (texlive-fonts-recommended)
set -e
cd "$(dirname "$0")"
pandoc combined_report.md \
  --pdf-engine=xelatex \
  --toc \
  --toc-depth=2 \
  --number-sections \
  -V toc-title="Contents" \
  -o combined_report.pdf
echo "Done: combined_report.pdf"
