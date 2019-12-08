#!/bin/bash

for args in --factors={{Lemmas,Lemmas\,Tags}\ --lemma_rule_min={1,2,3},Tags}{,\ --elmo=pdt/bert-pdt-3.0}{,\ --embeddings=pdt/forms.vectors-w5-d300-ns5.npz}; do
  qsub -q gpu-troja.q -l gpu=1,mem_free=16G,h_data=32G -j y withcuda ~/venvs/tf-1.12-gpu/bin/python3 morpho_tagger.py pdt/pdt-3.0 $args
done
