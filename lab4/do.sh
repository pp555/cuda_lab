#!/bin/bash

for i in seq 1 100 1000
do
./do.py 10 > tmp.unroll.cu
nvcc tmp.unroll.cu
./a.out
done
