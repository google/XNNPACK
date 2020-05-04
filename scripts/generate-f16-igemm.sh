#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

########################## ARM NEON with FP16 compute #########################
### LD64 micro-kernels
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=1 -D NR=8 -o src/f16-igemm/gen/1x8-minmax-neonfp16arith-ld64.c
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=4 -D NR=8 -o src/f16-igemm/gen/4x8-minmax-neonfp16arith-ld64.c
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=6 -D NR=8 -o src/f16-igemm/gen/6x8-minmax-neonfp16arith-ld64.c
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=8 -D NR=8 -o src/f16-igemm/gen/8x8-minmax-neonfp16arith-ld64.c

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/f16-igemm-minmax.yaml --output test/f16-igemm-minmax.cc
