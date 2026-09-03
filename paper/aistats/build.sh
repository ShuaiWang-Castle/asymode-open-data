#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 make_solvable_figure.py

if [[ ! -f aistats2026.sty ]]; then
  tmp="$(mktemp -d)"
  trap 'rm -rf "$tmp"' EXIT
  curl -fsSL "https://aistats.org/aistats2026/AISTATS2026PaperPack.zip" -o "$tmp/pack.zip" \
    || curl -fsSL "https://raw.githubusercontent.com/aistats/aistats2026/gh-pages/AISTATS2026PaperPack.zip" -o "$tmp/pack.zip"
  unzip -q "$tmp/pack.zip" -d "$tmp/pack"
  style="$(find "$tmp/pack" -name aistats2026.sty -print -quit)"
  [[ -n "$style" ]] || { echo "aistats2026.sty not found in official pack" >&2; exit 1; }
  cp "$style" ./aistats2026.sty
  fancy="$(find "$tmp/pack" -name fancyhdr.sty -print -quit)"
  [[ -z "$fancy" ]] || cp "$fancy" ./fancyhdr.sty
fi

latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

if grep -Eqi "undefined references|Citation .* undefined|There were undefined" main.log; then
  echo "Unresolved references or citations remain" >&2
  exit 1
fi

pdfinfo main.pdf | tee pdfinfo.txt
grep -q 'Page size:.*612 x 792 pts' pdfinfo.txt
pdffonts main.pdf | tee pdffonts.txt
if awk 'NR>2 && $2=="Type" && $3=="3" {found=1} END {exit found?0:1}' pdffonts.txt; then
  echo "Type 3 fonts detected" >&2
  exit 1
fi
