#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
### Microkernels without unrolling
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=8  -D NR=1 -D UNROLL=1 -o src/f16-spmm/gen/8x1-minmax-neonfp16arith.c
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=16 -D NR=1 -D UNROLL=1 -o src/f16-spmm/gen/16x1-minmax-neonfp16arith.c
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=24 -D NR=1 -D UNROLL=1 -o src/f16-spmm/gen/24x1-minmax-neonfp16arith.c
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=32 -D NR=1 -D UNROLL=1 -o src/f16-spmm/gen/32x1-minmax-neonfp16arith.c
### Microkernels with 2X unrolling
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=8  -D NR=1 -D UNROLL=2 -o src/f16-spmm/gen/8x1-minmax-neonfp16arith-unroll2.c
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=16 -D NR=1 -D UNROLL=2 -o src/f16-spmm/gen/16x1-minmax-neonfp16arith-unroll2.c
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=24 -D NR=1 -D UNROLL=2 -o src/f16-spmm/gen/24x1-minmax-neonfp16arith-unroll2.c
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=32 -D NR=1 -D UNROLL=2 -o src/f16-spmm/gen/32x1-minmax-neonfp16arith-unroll2.c

################################## Unit tests #################################
tools/generate-spmm-test.py --spec test/f16-spmm-minmax.yaml --output test/f16-spmm-minmax.cc
