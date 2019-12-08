# Scripts for running morphological analysis

- `analyse.sh`:
  Run morphological analysis using MorphoDiTa. Requires:
  - run_morpho_analyze binary from MorphoDita
  - morphological dictionary

  The input is a file in vertical format (word per line, empty line is end of
  sentence) with forms in the first column. The output is the content of the
  input file with the analyses appended to each line, as tab-separated lemma-tag
  pairs. Namely, all columns of the input file are kept.

  The analysis does not use a guesser and unlike MorphoDiTa it does not
  use `X@-------------` for unknown words, i.e., there are no analyses
  for unknown words.

- `enrich_from_train.py`:
  Enrich morphological analyses by using train data gold annotations.

  Sometimes there are inconsistencies between gold annotations data and the
  dictionary. For all word forms for which there are any morphological
  analyses from the dictionary, this script also adds all gold lemmas and tags
  present in the train data (this is equivalent to enlarging the dictionary
  by all gold train data, but only for forms already present in the dictionary).

  The input data should be in vertical format, first three columns are form,
  gold lemma and gold tag, followed by the tab-separated morphological analyses.
  The train data is assumed to be in the same format (but its morphological
  analyses, if present, are ignored).
