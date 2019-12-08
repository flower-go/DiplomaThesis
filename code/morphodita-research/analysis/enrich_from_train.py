#!/usr/bin/env python3
import argparse
import collections
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument("train_data", type=str, help="Training data file")
args = parser.parse_args()

lemma_strip_re = re.compile(r"(?<=.)(?:`|_|-[^0-9]).*$")
def lemma_tag_key(lemma, tag):
    return lemma_strip_re.sub("", lemma) + "\t" + tag

train = collections.defaultdict(lambda: dict())
with open(args.train_data, mode="r", encoding="utf-8") as train_file:
    for line in train_file:
        line = line.rstrip("\n")
        if not line: continue

        form, lemma, tag, *_ = line.split("\t")
        train[form][lemma_tag_key(lemma, tag)] = (lemma, tag)

for line in sys.stdin:
    line = line.rstrip("\n")

    if not line:
        print()
    else:
        form, lemma, tag, *analyses = line.split("\t")
        assert len(analyses) % 2 == 0

        if len(analyses):
            train_analyses = list(train.get(form, {}).items()) \
                + list(train.get(form[0] + form[1:].lower(), {}).items()) \
                + list(train.get(form.lower(), {}).items())
            if train_analyses:
                analyses_set = set(lemma_tag_key(analyses[i], analyses[i + 1]) for i in range(0, len(analyses), 2))
                for key, lemma_tag in train_analyses:
                    if key not in analyses_set:
                        analyses.extend(lemma_tag)
                        analyses_set.add(key)

        print("\t".join([form, lemma, tag] + analyses))
