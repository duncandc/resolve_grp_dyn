#!/bin/bash

python setup.py build_ext --inplace
cp ./pair_counter/pairwise_distance_rp_pi.so ./
rm -r ./pair_counter
rm -r ./build

