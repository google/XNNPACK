#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
### Microkernels without unrolling
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=8  -D NR=1 -D UNROLL=1 -o src/f16-spmm/gen/f16-spmm-8x1-minmax-neonfp16arith.c &
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=16 -D NR=1 -D UNROLL=1 -o src/f16-spmm/gen/f16-spmm-16x1-minmax-neonfp16arith.c &
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=24 -D NR=1 -D UNROLL=1 -o src/f16-spmm/gen/f16-spmm-24x1-minmax-neonfp16arith.c &
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=32 -D NR=1 -D UNROLL=1 -o src/f16-spmm/gen/f16-spmm-32x1-minmax-neonfp16arith.c &
### Microkernels with 2X unrolling
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=8  -D NR=1 -D UNROLL=2 -o src/f16-spmm/gen/f16-spmm-8x1-minmax-neonfp16arith-u2.c &
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=16 -D NR=1 -D UNROLL=2 -o src/f16-spmm/gen/f16-spmm-16x1-minmax-neonfp16arith-u2.c &
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=24 -D NR=1 -D UNROLL=2 -o src/f16-spmm/gen/f16-spmm-24x1-minmax-neonfp16arith-u2.c &
tools/xngen src/f16-spmm/neonfp16arith.c.in -D MR=32 -D NR=1 -D UNROLL=2 -o src/f16-spmm/gen/f16-spmm-32x1-minmax-neonfp16arith-u2.c &

### Microkernels with software pipelining
tools/xngen src/f16-spmm/neonfp16arith-pipelined.c.in -D MR=8  -D NR=1 -o src/f16-spmm/gen/f16-spmm-8x1-minmax-neonfp16arith-pipelined.c &
tools/xngen src/f16-spmm/neonfp16arith-pipelined.c.in -D MR=16 -D NR=1 -o src/f16-spmm/gen/f16-spmm-16x1-minmax-neonfp16arith-pipelined.c &
tools/xngen src/f16-spmm/neonfp16arith-pipelined.c.in -D MR=24 -D NR=1 -o src/f16-spmm/gen/f16-spmm-24x1-minmax-neonfp16arith-pipelined.c &
tools/xngen src/f16-spmm/neonfp16arith-pipelined.c.in -D MR=32 -D NR=1 -o src/f16-spmm/gen/f16-spmm-32x1-minmax-neonfp16arith-pipelined.c &

################################# RISC-V Vector #################################
tools/xngen src/f16-spmm/rvv.c.in -D MR=m1 -D NR=1 -o src/f16-spmm/gen/f16-spmm-1vx1-minmax-rvvfp16arith.c &
tools/xngen src/f16-spmm/rvv.c.in -D MR=m2 -D NR=1 -o src/f16-spmm/gen/f16-spmm-2vx1-minmax-rvvfp16arith.c &
tools/xngen src/f16-spmm/rvv.c.in -D MR=m4 -D NR=1 -o src/f16-spmm/gen/f16-spmm-4vx1-minmax-rvvfp16arith.c &
tools/xngen src/f16-spmm/rvv.c.in -D MR=m8 -D NR=1 -o src/f16-spmm/gen/f16-spmm-8vx1-minmax-rvvfp16arith.c &

wait
