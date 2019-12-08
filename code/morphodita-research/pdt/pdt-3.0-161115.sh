#!/bin/sh

for f in ../*.o*; do
  echo \
    $(awk '/^embeddings:/ {print "WE:" ($2 == "None" ? "No " : "Yes")}' $f) \
    $(awk '/^elmo:/ {print "Bert:" ($2 == "None" ? "No " : "Yes")}' $f) \
    $(awk '/^lemma_rule_min:/ {print "LRmin:" $2 " |:|"}' $f) \
    $(sed -n 's/^Test, epoch 20, lr 0.0001,//;T;s/: */:/g; s/,//g; p' $f)
done | python -c '
import sys

data = [["WE", "Bert", "LRmin", "|", "TagsRaw", "TagsDict", "LemmasRaw", "LemmasDict", "LemmasTagsRaw", "LemmasTagsDict"]]
for line in sys.stdin:
  line = line.rstrip("\n")
  data.append([""] * len(data[0]))
  for field in line.split():
    column, value = field.split(":")
    data[-1][data[0].index(column)] = value

widths = [max(map(len, values)) for values in zip(*data)]
for line in data:
  print(" ".join(value.ljust(width) for value, width in zip(line, widths)))
'
