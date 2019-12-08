#!/bin/sh

[ $# -ge 1 ] || { echo Usage: $0 MorphoDiTa_dictionary_file input_file \>output_file >&2; exit 1; }
dictionary="$1"; shift
input="$1"; shift
run_morpho_analyze="$(dirname $0)/run_morpho_analyze"

[ -x "$run_morpho_analyze" ] || { echo Missing $run_morpho_analyze from MorphoDiTa >&2; exit 1; }

cut -f1 "$input" | "$run_morpho_analyze" "$dictionary" 0 --input=vertical --output=vertical \
  | sed 's/^[^\t]*\t\?//' | paste "$input" - | sed 's/^\t$//; s/\t[^\t]*\tX@-------------$//'
