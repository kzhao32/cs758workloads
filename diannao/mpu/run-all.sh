#!/bin/bash

for trial in "1" "2"; do
  args="no run"
  if [ "$trial" = "1" ]; then
    args="perf"
  fi
  for i in conv1p conv2p conv3p conv4p conv5p pool1p pool3p pool5p class1p class3p; do
    echo -n "$i $args "
    out=`$i $args`
    ticks=`echo $out | cut -d: -f2`
    echo $ticks
  done
done
