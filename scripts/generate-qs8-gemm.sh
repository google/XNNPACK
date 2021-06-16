#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -o src/qs8-gemm/gen/1x2-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -o src/qs8-gemm/gen/2x2-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=2 -o src/qs8-gemm/gen/3x2-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=2 -o src/qs8-gemm/gen/4x2-minmax-gemmlowp-scalar.c

tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -o src/qs8-gemm/gen/1x4-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -o src/qs8-gemm/gen/2x4-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-gemm/scalar.c.in -D MR=3 -D NR=4 -o src/qs8-gemm/gen/3x4-minmax-gemmlowp-scalar.c
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -o src/qs8-gemm/gen/4x4-minmax-gemmlowp-scalar.c

################################## WAsm SIMD ##################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-wasmsimd-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-wasmsimd-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-wasmsimd-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-wasmsimd-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-wasmsimd-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-wasmsimd-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-gemmlowp-wasmsimd.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-gemmlowp-wasmsimd.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-gemmlowp-wasmsimd.c

################################### ARM NEON ##################################
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=0 -o src/qs8-gemm/gen/1x8-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=0 -o src/qs8-gemm/gen/2x8-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=0 -o src/qs8-gemm/gen/3x8-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=0 -o src/qs8-gemm/gen/4x8-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=0 -o src/qs8-gemm/gen/6x8-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=0 -o src/qs8-gemm/gen/1x16-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=0 -o src/qs8-gemm/gen/2x16-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=0 -o src/qs8-gemm/gen/3x16-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=0 -o src/qs8-gemm/gen/4x16-minmax-gemmlowp-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=0 -o src/qs8-gemm/gen/6x16-minmax-gemmlowp-neon-mlal-lane.c

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8  -D PREFETCH=1 -o src/qs8-gemm/gen/1x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8  -D PREFETCH=1 -o src/qs8-gemm/gen/2x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8  -D PREFETCH=1 -o src/qs8-gemm/gen/3x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8  -D PREFETCH=1 -o src/qs8-gemm/gen/4x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=8  -D PREFETCH=1 -o src/qs8-gemm/gen/6x8-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -D PREFETCH=1 -o src/qs8-gemm/gen/1x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -D PREFETCH=1 -o src/qs8-gemm/gen/2x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -D PREFETCH=1 -o src/qs8-gemm/gen/3x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -D PREFETCH=1 -o src/qs8-gemm/gen/4x16-minmax-gemmlowp-neon-mlal-lane-prfm.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=6 -D NR=16 -D PREFETCH=1 -o src/qs8-gemm/gen/6x16-minmax-gemmlowp-neon-mlal-lane-prfm.c

tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=1 -D NR=8 -o src/qs8-gemm/gen/1x8-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=2 -D NR=8 -o src/qs8-gemm/gen/2x8-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=3 -D NR=8 -o src/qs8-gemm/gen/3x8-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=4 -D NR=8 -o src/qs8-gemm/gen/4x8-minmax-gemmlowp-neon-mull-addw-dup.c

tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=1 -D NR=16 -o src/qs8-gemm/gen/1x16-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=2 -D NR=16 -o src/qs8-gemm/gen/2x16-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=3 -D NR=16 -o src/qs8-gemm/gen/3x16-minmax-gemmlowp-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=4 -D NR=16 -o src/qs8-gemm/gen/4x16-minmax-gemmlowp-neon-mull-addw-dup.c

### C2 micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/1x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/2x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/3x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/4x8c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/1x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/2x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/3x16c2-minmax-gemmlowp-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/4x16c2-minmax-gemmlowp-neon-mull-padal-dup.c

tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/1x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/2x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/3x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/4x8c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/1x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/2x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/3x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/4x16c2-minmax-gemmlowp-neon-mlal-padal-dup.c

### C8 micro-kernels
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/3x8c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/4x8c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/1x16c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/2x16c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/3x16c8-minmax-gemmlowp-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/4x16c8-minmax-gemmlowp-neon-mull-padal.c

tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/3x8c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/4x8c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/1x16c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/2x16c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/3x16c8-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/4x16c8-minmax-gemmlowp-neon-mlal-padal.c

### C16 micro-kernels
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=1 -D NR=8  -o src/qs8-gemm/gen/1x8c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=2 -D NR=8  -o src/qs8-gemm/gen/2x8c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=3 -D NR=8  -o src/qs8-gemm/gen/3x8c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=4 -D NR=8  -o src/qs8-gemm/gen/4x8c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=1 -D NR=16 -o src/qs8-gemm/gen/1x16c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=2 -D NR=16 -o src/qs8-gemm/gen/2x16c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=3 -D NR=16 -o src/qs8-gemm/gen/3x16c16-minmax-gemmlowp-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=4 -D NR=16 -o src/qs8-gemm/gen/4x16c16-minmax-gemmlowp-neon-mlal-padal.c

### C4 micro-kernels
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/1x8c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/4x8c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/6x8c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/8x8c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/1x16c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/4x16c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/6x16c4-minmax-gemmlowp-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/8x16c4-minmax-gemmlowp-neondot.c

tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=1  -D NR=8  -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/1x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=4  -D NR=8  -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/4x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=6  -D NR=8  -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/6x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=8  -D NR=8  -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/8x8c4-minmax-fp32-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=1  -D NR=16 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/1x16c4-minmax-fp32-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=4  -D NR=16 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/4x16c4-minmax-fp32-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=6  -D NR=16 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/6x16c4-minmax-fp32-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=8  -D NR=16 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/8x16c4-minmax-fp32-neondot.c

############################### AArch64 assembly ##############################
# Cortex A53 micro-kernel
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=0 -o src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-cortex-a53.S
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=1 -o src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm-cortex-a53.S
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=0 -o src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-cortex-a53.S
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-padal-cortex-a53.S.in -D PREFETCH=1 -o src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm-cortex-a53.S
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in   -D PREFETCH=0 -o src/qs8-gemm/gen/4x16-minmax-gemmlowp-aarch64-neon-mlal-lane-cortex-a53.S
tools/xngen src/qs8-gemm/4x16-aarch64-neon-mlal-lane-cortex-a53.S.in   -D PREFETCH=1 -o src/qs8-gemm/gen/4x16-minmax-gemmlowp-aarch64-neon-mlal-lane-prfm-cortex-a53.S

# Cortex A75 micro-kernel
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-padal.S.in -D PREFETCH=0 -o src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal.S
tools/xngen src/qs8-gemm/1x8c8-aarch64-neon-mlal-padal.S.in -D PREFETCH=1 -o src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm.S
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-padal.S.in -D PREFETCH=0 -o src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal.S
tools/xngen src/qs8-gemm/2x8c8-aarch64-neon-mlal-padal.S.in -D PREFETCH=1 -o src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mlal-padal-prfm.S

tools/xngen src/qs8-gemm/1x16c4-minmax-gemmlowp-aarch64-neondot-ld32.S.in       -o src/qs8-gemm/gen/1x16c4-minmax-gemmlowp-aarch64-neondot-ld32.S
tools/xngen src/qs8-gemm/1x16c4-minmax-gemmlowp-aarch64-neondot-ld64.S.in       -o src/qs8-gemm/gen/1x16c4-minmax-gemmlowp-aarch64-neondot-ld64.S
tools/xngen src/qs8-gemm/2x8c16-minmax-gemmlowp-aarch64-neon-mlal-padal.S.in    -o src/qs8-gemm/gen/2x8c16-minmax-gemmlowp-aarch64-neon-mlal-padal.S
tools/xngen src/qs8-gemm/2x8c8-minmax-gemmlowp-aarch64-neon-mull-padal.S.in     -o src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-aarch64-neon-mull-padal.S
tools/xngen src/qs8-gemm/4x16c4-minmax-gemmlowp-aarch64-neondot-cortex-a55.S.in -o src/qs8-gemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-cortex-a55.S
tools/xngen src/qs8-gemm/4x16c4-minmax-gemmlowp-aarch64-neondot-ld32.S.in       -o src/qs8-gemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-ld32.S
tools/xngen src/qs8-gemm/4x16c4-minmax-gemmlowp-aarch64-neondot-ld64.S.in       -o src/qs8-gemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-ld64.S
tools/xngen src/qs8-gemm/4x16c4-minmax-gemmlowp-aarch64-neondot-ld128.S.in      -o src/qs8-gemm/gen/4x16c4-minmax-gemmlowp-aarch64-neondot-ld128.S

################################### x86 SSE ###################################
### C2 micro-kernels
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-sse2-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-fp32-sse2-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2-minmax-fp32-sse2-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-ssse3-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-ssse3-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-ssse3-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-ssse3-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-sse41-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-fp32-sse41-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2-minmax-fp32-sse41-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-avx-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-fp32-avx-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2-minmax-fp32-avx-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-xop-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-fp32-xop-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c2-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/4x4c2-minmax-fp32-xop-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-sse2-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-fp32-sse2-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2-minmax-fp32-sse2-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-ssse3-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-ssse3-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-ssse3-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-ssse3-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-sse41-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-fp32-sse41-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2-minmax-fp32-sse41-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-avx-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-fp32-avx-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2-minmax-fp32-avx-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-gemmlowp-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-gemmlowp-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-gemmlowp-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-gemmlowp-xop-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-fp32-xop-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c2-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/4x4c2-minmax-fp32-xop-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-gemmlowp-sse2.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2-xw-minmax-gemmlowp-sse2.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2-xw-minmax-gemmlowp-sse2.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-gemmlowp-sse2.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-gemmlowp-ssse3.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2-xw-minmax-gemmlowp-ssse3.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2-xw-minmax-gemmlowp-ssse3.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-gemmlowp-ssse3.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-gemmlowp-sse41.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2-xw-minmax-gemmlowp-sse41.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2-xw-minmax-gemmlowp-sse41.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-gemmlowp-sse41.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-gemmlowp-avx.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2-xw-minmax-gemmlowp-avx.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2-xw-minmax-gemmlowp-avx.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-gemmlowp-avx.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-gemmlowp-xop.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c2-xw-minmax-gemmlowp-xop.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c2-xw-minmax-gemmlowp-xop.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-gemmlowp-xop.c

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-sse2-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c8-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c8-minmax-fp32-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c8-minmax-fp32-sse2-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-ssse3-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-ssse3-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-ssse3-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-fp32-ssse3-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-fp32-ssse3-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-fp32-ssse3-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-sse41-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c8-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c8-minmax-fp32-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c8-minmax-fp32-sse41-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-avx-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-fp32-avx-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c8-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c8-minmax-fp32-avx-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c8-minmax-fp32-avx-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-xop-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-fp32-xop-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/1x4c8-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/2x4c8-minmax-fp32-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD64     -o src/qc8-gemm/gen/3x4c8-minmax-fp32-xop-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-sse2-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-fp32-sse2-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c8-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c8-minmax-fp32-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c8-minmax-fp32-sse2-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-ssse3-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-ssse3-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-ssse3-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-fp32-ssse3-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-fp32-ssse3-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-fp32-ssse3-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-sse41-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-fp32-sse41-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c8-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c8-minmax-fp32-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c8-minmax-fp32-sse41-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-avx-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-fp32-avx-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c8-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c8-minmax-fp32-avx-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c8-minmax-fp32-avx-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-gemmlowp-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-gemmlowp-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-gemmlowp-xop-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-fp32-xop-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/1x4c8-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/2x4c8-minmax-fp32-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -D VARIANT=LD128    -o src/qc8-gemm/gen/3x4c8-minmax-fp32-xop-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-gemmlowp-sse2.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-gemmlowp-sse2.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-gemmlowp-sse2.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-gemmlowp-ssse3.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-gemmlowp-ssse3.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-gemmlowp-ssse3.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-gemmlowp-sse41.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-gemmlowp-sse41.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=0 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-gemmlowp-sse41.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-gemmlowp-avx.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-gemmlowp-avx.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=0 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-gemmlowp-avx.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-gemmlowp-xop.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-gemmlowp-xop.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D AVX=1 -D XOP=1 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-gemmlowp-xop.c

################################### x86 AVX2 ##################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=LD128    -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/1x8c8-minmax-gemmlowp-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=LD128    -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/2x8c8-minmax-gemmlowp-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=LD128    -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/3x8c8-minmax-gemmlowp-avx2.c

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=LD128    -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/1x8c8-minmax-fp32-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=LD128    -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/2x8c8-minmax-fp32-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=LD128    -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/3x8c8-minmax-fp32-avx2.c

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=LD128    -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/1x8c8-minmax-fp32-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=LD128    -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/2x8c8-minmax-fp32-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=LD128    -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/3x8c8-minmax-fp32-avx2.c

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=EXTENDED -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/1x8c8-xw-minmax-gemmlowp-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=EXTENDED -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/2x8c8-xw-minmax-gemmlowp-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=EXTENDED -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/3x8c8-xw-minmax-gemmlowp-avx2.c

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=EXTENDED -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/1x8c8-xw-minmax-fp32-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=EXTENDED -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/2x8c8-xw-minmax-fp32-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=EXTENDED -D CHANNELWISE=0 -D REQUANTIZATION=FP32     -o src/qs8-gemm/gen/3x8c8-xw-minmax-fp32-avx2.c

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=EXTENDED -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/1x8c8-xw-minmax-fp32-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=EXTENDED -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/2x8c8-xw-minmax-fp32-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=EXTENDED -D CHANNELWISE=1 -D REQUANTIZATION=FP32     -o src/qc8-gemm/gen/3x8c8-xw-minmax-fp32-avx2.c

################################## x86 AVX512 #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/1x16c8-minmax-gemmlowp-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/2x16c8-minmax-gemmlowp-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/3x16c8-minmax-gemmlowp-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=GEMMLOWP -o src/qs8-gemm/gen/4x16c8-minmax-gemmlowp-avx512skx.c

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=FP32 -o src/qs8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=FP32 -o src/qs8-gemm/gen/2x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=FP32 -o src/qs8-gemm/gen/3x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D VARIANT=LD256 -D CHANNELWISE=0 -D REQUANTIZATION=FP32 -o src/qs8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c

tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D VARIANT=LD256 -D CHANNELWISE=1 -D REQUANTIZATION=FP32 -o src/qc8-gemm/gen/1x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D VARIANT=LD256 -D CHANNELWISE=1 -D REQUANTIZATION=FP32 -o src/qc8-gemm/gen/2x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D VARIANT=LD256 -D CHANNELWISE=1 -D REQUANTIZATION=FP32 -o src/qc8-gemm/gen/3x16c8-minmax-fp32-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D VARIANT=LD256 -D CHANNELWISE=1 -D REQUANTIZATION=FP32 -o src/qc8-gemm/gen/4x16c8-minmax-fp32-avx512skx.c

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/qs8-gemm-minmax-gemmlowp.yaml --output test/qs8-gemm-minmax-gemmlowp.cc
tools/generate-gemm-test.py --spec test/qs8-gemm-minmax-fp32.yaml --output test/qs8-gemm-minmax-fp32.cc
tools/generate-gemm-test.py --spec test/qc8-gemm-minmax-fp32.yaml --output test/qc8-gemm-minmax-fp32.cc
