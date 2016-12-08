#!/bin/bash
make
#for size in "256" "1024" "4096" "16384"; do
for size in "128" "512" "2048" "8192"; do
    for numProcs in "1" "2" "4" "8" "16" "32" "64" "128" "256"; do
        for numTrial in "1" "2" "3" "4" "5"; do
            out=`./sift-dog -x $size -y $size -p $numProcs`
            echo $out
        done
    done
done
