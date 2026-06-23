#!/usr/bin/env bash
# Rebuild missed_thrust_onoff_report.pdf from the markdown source.
# Requires: pandoc >= 3, xelatex (texlive-xetex), TeX Gyre fonts (texlive-fonts-recommended)
set -e
cd "$(dirname "$0")"
pandoc missed_thrust_onoff_report.md \
  --pdf-engine=xelatex \
  --toc \
  --toc-depth=2 \
  --number-sections \
  -V toc-title="Contents" \
  -o missed_thrust_onoff_report.pdf
echo "Done: missed_thrust_onoff_report.pdf"
