#!/bin/sh

for dataset in dev test; do
  sh ../analysis/analyse.sh dictionary/czech-morfflex-161115.dict pdt-3.0-$dataset.ori \
    | python3 ../analysis/enrich_from_train.py pdt-3.0-train.txt >pdt-3.0-$dataset.txt
done
