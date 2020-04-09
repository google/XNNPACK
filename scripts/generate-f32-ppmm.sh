#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-ppmm/scalar.c.in -D MR=4 -D NR=4 -o src/f32-ppmm/gen/4x4-minmax-scalar.c
tools/xngen src/f32-ppmm/scalar.c.in -D MR=2 -D NR=4 -o src/f32-ppmm/gen/2x4-minmax-scalar.c
tools/xngen src/f32-ppmm/scalar.c.in -D MR=4 -D NR=2 -o src/f32-ppmm/gen/4x2-minmax-scalar.c
tools/xngen src/f32-ppmm/scalar.c.in -D MR=3 -D NR=3 -o src/f32-ppmm/gen/3x3-minmax-scalar.c

################################### ARM NEON ##################################
tools/xngen src/f32-ppmm/neon.c.in -D MR=4 -D NR=8 -D FMA=0 -o src/f32-ppmm/gen/4x8-minmax-neon.c
tools/xngen src/f32-ppmm/neon.c.in -D MR=4 -D NR=8 -D FMA=1 -o src/f32-ppmm/gen/4x8-minmax-neonfma.c
tools/xngen src/f32-ppmm/neon.c.in -D MR=8 -D NR=8 -D FMA=0 -o src/f32-ppmm/gen/8x8-minmax-neon.c
tools/xngen src/f32-ppmm/neon.c.in -D MR=8 -D NR=8 -D FMA=1 -o src/f32-ppmm/gen/8x8-minmax-neonfma.c

#################################### PSIMD ####################################
tools/xngen src/f32-ppmm/psimd.c.in -D MR=4 -D NR=8 -o src/f32-ppmm/gen/4x8-minmax-psimd.c

################################### x86 SSE ###################################
tools/xngen src/f32-ppmm/sse.c.in -D MR=4 -D NR=8 -o src/f32-ppmm/gen/4x8-minmax-sse.c

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/f32-ppmm-minmax.yaml --output test/f32-ppmm-minmax.cc
