#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Microkernels without unrolling
tools/xngen src/f32-spmm/scalar.c.in -D MR=1 -D NR=1 -D UNROLL=1 -o src/f32-spmm/gen/1x1-scalar.c
tools/xngen src/f32-spmm/scalar.c.in -D MR=2 -D NR=1 -D UNROLL=1 -o src/f32-spmm/gen/2x1-scalar.c
tools/xngen src/f32-spmm/scalar.c.in -D MR=4 -D NR=1 -D UNROLL=1 -o src/f32-spmm/gen/4x1-scalar.c
tools/xngen src/f32-spmm/scalar.c.in -D MR=8 -D NR=1 -D UNROLL=1 -o src/f32-spmm/gen/8x1-scalar.c
tools/xngen src/f32-spmm/scalar.c.in -D MR=8 -D NR=2 -D UNROLL=1 -o src/f32-spmm/gen/8x2-scalar.c
tools/xngen src/f32-spmm/scalar.c.in -D MR=8 -D NR=4 -D UNROLL=1 -o src/f32-spmm/gen/8x4-scalar.c
### Microkernels with software pipelining
tools/xngen src/f32-spmm/scalar-pipelined.c.in -D MR=1 -D NR=1 -o src/f32-spmm/gen/1x1-scalar-pipelined.c
tools/xngen src/f32-spmm/scalar-pipelined.c.in -D MR=2 -D NR=1 -o src/f32-spmm/gen/2x1-scalar-pipelined.c
tools/xngen src/f32-spmm/scalar-pipelined.c.in -D MR=4 -D NR=1 -o src/f32-spmm/gen/4x1-scalar-pipelined.c
tools/xngen src/f32-spmm/scalar-pipelined.c.in -D MR=8 -D NR=1 -o src/f32-spmm/gen/8x1-scalar-pipelined.c

################################### ARM NEON ##################################
### Microkernels without unrolling
tools/xngen src/f32-spmm/neon.c.in -D MR=4 -D NR=1 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/4x1-neonfma.c
tools/xngen src/f32-spmm/neon.c.in -D MR=8 -D NR=1 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/8x1-neonfma.c
tools/xngen src/f32-spmm/neon.c.in -D MR=12 -D NR=1 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/12x1-neonfma.c
tools/xngen src/f32-spmm/neon.c.in -D MR=16 -D NR=1 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/16x1-neonfma.c
### Microkernels with 2X unrolling
tools/xngen src/f32-spmm/neon.c.in -D MR=4  -D NR=1 -D UNROLL=2 -D FMA=1 -o src/f32-spmm/gen/4x1-neonfma-unroll2.c
tools/xngen src/f32-spmm/neon.c.in -D MR=8  -D NR=1 -D UNROLL=2 -D FMA=1 -o src/f32-spmm/gen/8x1-neonfma-unroll2.c
tools/xngen src/f32-spmm/neon.c.in -D MR=16  -D NR=1 -D UNROLL=2 -D FMA=1 -o src/f32-spmm/gen/16x1-neonfma-unroll2.c
### Microkernels for blocks of several output channels
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=4 -D NR=2 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/4x2-neonfma.c
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=8 -D NR=2 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/8x2-neonfma.c
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=12 -D NR=2 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/12x2-neonfma.c
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=16 -D NR=2 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/16x2-neonfma.c
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=4 -D NR=4 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/4x4-neonfma.c
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=8 -D NR=4 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/8x4-neonfma.c
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=12 -D NR=4 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/12x4-neonfma.c
tools/xngen src/f32-spmm/neon-blocked.c.in -D MR=16 -D NR=4 -D UNROLL=1 -D FMA=1 -o src/f32-spmm/gen/16x4-neonfma.c
### Microkernels with software pipelining
tools/xngen src/f32-spmm/neon-pipelined.c.in -D MR=4 -D NR=1 -D FMA=1 -o src/f32-spmm/gen/4x1-neonfma-pipelined.c
tools/xngen src/f32-spmm/neon-pipelined.c.in -D MR=8 -D NR=1 -D FMA=1 -o src/f32-spmm/gen/8x1-neonfma-pipelined.c
tools/xngen src/f32-spmm/neon-pipelined.c.in -D MR=16 -D NR=1 -D FMA=1 -o src/f32-spmm/gen/16x1-neonfma-pipelined.c

################################### x86 SSE ###################################
### Microkernels without unrolling
tools/xngen src/f32-spmm/sse.c.in -D MR=4 -D NR=1 -D UNROLL=1 -o src/f32-spmm/gen/4x1-sse.c
tools/xngen src/f32-spmm/sse.c.in -D MR=8 -D NR=1 -D UNROLL=1 -o src/f32-spmm/gen/8x1-sse.c

################################## Unit tests #################################
tools/generate-spmm-test.py --spec test/f32-spmm.yaml --output test/f32-spmm.cc
