# Experimental POS Tagging and Lemmatization

This repository contains experimental prototype for POS tagging and
lemmatization, either without or with a morphological dictionary.

The script requires Python 3 and Tensorflow, versions 1.5-1.13 are supported
(even if 1.13 prints out a lot of "obsolete" warnings).


## Input Data

The input data are assumed to be in vertical format, word on a line,
with empty line being end of sentence. Each line consists of tab-separated
columns. The first three are `form`, `gold_lemma`, `gold_tag`. The
rest of the columns are optional and may contain morphological analyses
as tab-separated `lemma`, `tag` pairs.

The morphological analyses for train data is not used in any way, the
morphological analyses are utilized only for prediction. Furthermore,
even if predicting, the input file must have three columns, with `gold_lemma`
and `gold_tag` present (usually `_` is used).

Note that various comments are stripped from lemmas (the `lemma_re_strip` regular
expression specifies what exactly to strip), so that only the raw lemma and
possibly a number is kept (i.e., `moc-1`). During prediction, if morphological
analyses are present, they are disambiguated using these stripped lemmas, but the
full lemma is printed on the output.


## Training

The training is performed by running
```
python3 morpho_tagger.py [options] input_data
```

Three input data files are loaded
- `${input_data}-train.txt`: required to exist, does not need morphological
  analyses (they are ignored)
- `${input_data}-dev.txt`: optional, used to measure performance (possibly
  including morphological analyses) after each epoch of training
- `${input_data}-test.txt`: optional, used to measure performance once after the
  training finishes

The training logs and the model are stored in a directory with a complicated
name (containing timestamp and all hyperparameter values) inside the `models`
subdirectory. After a successful training, it is a good idea to rename it :-)

If development or testing data are provided, the performance is measured
(accuracies of predicting lemmas, tags, and lemma-tag pairs, both without
the morphological analyses [`Raw`] and with them [`Dict`]) and stored in
TensorBoard logs and in `log` file in the model directory.

Important options:
- `--factors` (default `Lemmas,Tags`): which columns to predict, can be either
  `Lemmas`, `Tags` or `Lemmas,Tags`
- `--lemma_re_strip` (default ``r"(?<=.)(?:`|_|-[^0-9]).*$"``): a regular
  expression which is stripped from lemmas for disambiguation
- `--embeddings` (default `None`): optional path to a file with word embeddings.
  The word embeddings are assumed to be in `npz` format, with two fields
  - `words`: a Python list with word strings
  - `embeddings`: a Numpy array of shape `[#num_words, embedding_dimension]`,
  can be generated from `.vec` files by `embeddings/convert_vec_to_npz.py`
- `--elmo` (default `None`): precomputed contextualized embeddings, in the
  format generated by `embeddings/bert_conllu_embeddings.py`
- `--threads` (default 4): number of CPU threads to use (at most one GPU is used
  independently on this value)
- `--epochs` (default `40:1e-3,20:1e-4`): number of training epochs and associated
  learning rates

Other options are described in the `morpho_tagger.py` file.


## Prediction with a Trained Model

After a model is trained, it can be utilized by running
```
python3 morpho_tagger.py --predict=path_to_model_directory input_file
```
The generated content (input file with second and third columns predicted by the
model [or only one of them, depending on `--factor` settings]) is written
to standard output.

During prediction, all unspecified options are taken from the training. Notably
this includes `threads` and `embeddings`. If you would like different defaults
of the options for the trained models, you can override them in `options.json`
file in the model directory.
