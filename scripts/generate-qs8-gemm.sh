#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## WAsm SIMD ##################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-wasmsimd-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-wasmsimd-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-wasmsimd-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-wasmsimd-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-wasmsimd-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-wasmsimd-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=1 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-wasmsimd.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=2 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-wasmsimd.c
tools/xngen src/qs8-gemm/MRx4c8-wasmsimd.c.in -D MR=3 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-wasmsimd.c

################################### ARM NEON ##################################
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=8 -o src/qs8-gemm/gen/1x8-minmax-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=8 -o src/qs8-gemm/gen/2x8-minmax-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=8 -o src/qs8-gemm/gen/3x8-minmax-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=8 -o src/qs8-gemm/gen/4x8-minmax-neon-mlal-lane.c

tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=1 -D NR=16 -o src/qs8-gemm/gen/1x16-minmax-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=2 -D NR=16 -o src/qs8-gemm/gen/2x16-minmax-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=3 -D NR=16 -o src/qs8-gemm/gen/3x16-minmax-neon-mlal-lane.c
tools/xngen src/qs8-gemm/neon-mlal-lane.c.in -D MR=4 -D NR=16 -o src/qs8-gemm/gen/4x16-minmax-neon-mlal-lane.c

tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=1 -D NR=8 -o src/qs8-gemm/gen/1x8-minmax-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=2 -D NR=8 -o src/qs8-gemm/gen/2x8-minmax-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=3 -D NR=8 -o src/qs8-gemm/gen/3x8-minmax-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=4 -D NR=8 -o src/qs8-gemm/gen/4x8-minmax-neon-mull-addw-dup.c

tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=1 -D NR=16 -o src/qs8-gemm/gen/1x16-minmax-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=2 -D NR=16 -o src/qs8-gemm/gen/2x16-minmax-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=3 -D NR=16 -o src/qs8-gemm/gen/3x16-minmax-neon-mull-addw-dup.c
tools/xngen src/qs8-gemm/neon-mull-addw-dup.c.in -D MR=4 -D NR=16 -o src/qs8-gemm/gen/4x16-minmax-neon-mull-addw-dup.c

### C2 micro-kernels
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/1x8c2-minmax-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/2x8c2-minmax-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/3x8c2-minmax-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/4x8c2-minmax-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/1x16c2-minmax-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/2x16c2-minmax-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/3x16c2-minmax-neon-mull-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/4x16c2-minmax-neon-mull-padal-dup.c

tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/1x8c2-minmax-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/2x8c2-minmax-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/3x8c2-minmax-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/4x8c2-minmax-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=1 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/1x16c2-minmax-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=2 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/2x16c2-minmax-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=3 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/3x16c2-minmax-neon-mlal-padal-dup.c
tools/xngen src/qs8-gemm/c2-neon-mull-padal-dup.c.in -D MR=4 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/4x16c2-minmax-neon-mlal-padal-dup.c

### C8 micro-kernels
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/1x8c8-minmax-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/2x8c8-minmax-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/3x8c8-minmax-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=8  -D MLA=0 -o src/qs8-gemm/gen/4x8c8-minmax-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/1x16c8-minmax-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/2x16c8-minmax-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/3x16c8-minmax-neon-mull-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=16 -D MLA=0 -o src/qs8-gemm/gen/4x16c8-minmax-neon-mull-padal.c

tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/1x8c8-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/2x8c8-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/3x8c8-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=8  -D MLA=1 -o src/qs8-gemm/gen/4x8c8-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=1 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/1x16c8-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=2 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/2x16c8-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=3 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/3x16c8-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c8-neon-mull-padal.c.in -D MR=4 -D NR=16 -D MLA=1 -o src/qs8-gemm/gen/4x16c8-minmax-neon-mlal-padal.c

### C16 micro-kernels
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=1 -D NR=8  -o src/qs8-gemm/gen/1x8c16-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=2 -D NR=8  -o src/qs8-gemm/gen/2x8c16-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=3 -D NR=8  -o src/qs8-gemm/gen/3x8c16-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=4 -D NR=8  -o src/qs8-gemm/gen/4x8c16-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=1 -D NR=16 -o src/qs8-gemm/gen/1x16c16-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=2 -D NR=16 -o src/qs8-gemm/gen/2x16c16-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=3 -D NR=16 -o src/qs8-gemm/gen/3x16c16-minmax-neon-mlal-padal.c
tools/xngen src/qs8-gemm/c16-neon-mlal-padal.c.in -D MR=4 -D NR=16 -o src/qs8-gemm/gen/4x16c16-minmax-neon-mlal-padal.c

### C4 micro-kernels
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=1 -D NR=8  -o src/qs8-gemm/gen/1x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=2 -D NR=8  -o src/qs8-gemm/gen/2x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=3 -D NR=8  -o src/qs8-gemm/gen/3x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=4 -D NR=8  -o src/qs8-gemm/gen/4x8c4-minmax-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=1 -D NR=16 -o src/qs8-gemm/gen/1x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=2 -D NR=16 -o src/qs8-gemm/gen/2x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=3 -D NR=16 -o src/qs8-gemm/gen/3x16c4-minmax-neondot.c
tools/xngen src/qs8-gemm/MRxNRc4-neondot.c.in -D MR=4 -D NR=16 -o src/qs8-gemm/gen/4x16c4-minmax-neondot.c

################################### x86 SSE ###################################
### C2 micro-kernels
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D VARIANT=LD64  -o src/qs8-gemm/gen/1x4c2-minmax-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-sse2-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=3 -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-ssse3-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=3 -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-ssse3-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-sse41-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=5 -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c2-minmax-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=5 -D VARIANT=LD64     -o src/qs8-gemm/gen/4x4c2-minmax-xop-ld64.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-sse2-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=3 -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-ssse3-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=3 -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-ssse3-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-sse41-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=5 -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c2-minmax-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=5 -D VARIANT=LD128    -o src/qs8-gemm/gen/4x4c2-minmax-xop-ld128.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=2 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-sse2.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=2 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-sse2.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=3 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-ssse3.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=3 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-ssse3.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=4 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-sse41.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=4 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-sse41.c

tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=1 -D SSE=5 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c2-xw-minmax-xop.c
tools/xngen src/qs8-gemm/MRx4c2-sse.c.in -D MR=4 -D SSE=5 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/4x4c2-xw-minmax-xop.c

### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-sse2-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-sse2-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-ssse3-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-ssse3-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-ssse3-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-sse41-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-sse41-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=5 -D VARIANT=LD64     -o src/qs8-gemm/gen/1x4c8-minmax-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=5 -D VARIANT=LD64     -o src/qs8-gemm/gen/2x4c8-minmax-xop-ld64.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=5 -D VARIANT=LD64     -o src/qs8-gemm/gen/3x4c8-minmax-xop-ld64.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-sse2-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-sse2-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-ssse3-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-ssse3-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-ssse3-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-sse41-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-sse41-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=5 -D VARIANT=LD128    -o src/qs8-gemm/gen/1x4c8-minmax-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=5 -D VARIANT=LD128    -o src/qs8-gemm/gen/2x4c8-minmax-xop-ld128.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=5 -D VARIANT=LD128    -o src/qs8-gemm/gen/3x4c8-minmax-xop-ld128.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=2 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-sse2.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=2 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-sse2.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=2 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-sse2.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=3 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-ssse3.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=3 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-ssse3.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=3 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-ssse3.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=4 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-sse41.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=4 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-sse41.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=4 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-sse41.c

tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=1 -D SSE=5 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x4c8-xw-minmax-xop.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=2 -D SSE=5 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x4c8-xw-minmax-xop.c
tools/xngen src/qs8-gemm/MRx4c8-sse.c.in -D MR=3 -D SSE=5 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x4c8-xw-minmax-xop.c

################################### x86 AVX2 ##################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=LD128    -o src/qs8-gemm/gen/1x8c8-minmax-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=LD128    -o src/qs8-gemm/gen/2x8c8-minmax-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=LD128    -o src/qs8-gemm/gen/3x8c8-minmax-avx2.c

tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=1 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/1x8c8-xw-minmax-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=2 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/2x8c8-xw-minmax-avx2.c
tools/xngen src/qs8-gemm/MRx8c8-avx2.c.in -D MR=3 -D VARIANT=EXTENDED -o src/qs8-gemm/gen/3x8c8-xw-minmax-avx2.c

################################## x86 AVX512 #################################
### C8 micro-kernels
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=1 -D VARIANT=LD256    -o src/qs8-gemm/gen/1x16c8-minmax-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=2 -D VARIANT=LD256    -o src/qs8-gemm/gen/2x16c8-minmax-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=3 -D VARIANT=LD256    -o src/qs8-gemm/gen/3x16c8-minmax-avx512skx.c
tools/xngen src/qs8-gemm/MRx16c8-avx512skx.c.in -D MR=4 -D VARIANT=LD256    -o src/qs8-gemm/gen/4x16c8-minmax-avx512skx.c

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/qs8-gemm-minmax.yaml --output test/qs8-gemm-minmax.cc
