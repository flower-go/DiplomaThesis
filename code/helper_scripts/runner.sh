#!/bin/sh

[ $# -ge 1 ] || { echo Usage: $0 basename >&2; exit 1; }
basename="$1"
[ -f "$basename.limit" ] || { echo Expected "$basename.limit" to exist >&2; exit 1; }
[ -f "$basename.commands" ] || { echo Expected "$basename.commands" to exist >&2; exit 1; }

i=1;
while [ $i -le $(wc -l <"$basename.commands") ]; do
  if [ $(expr $(qstat | wc -l) - 2) -lt $(cat $basename.limit) ]; then
    cmd=$(sed -n "$i p" "$basename.commands")
    eval $cmd
    i=$(expr $i + 1)
  else
    sleep 5
  fi
done
