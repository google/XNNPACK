#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

############################### AArch64 assembly ##############################
### LD64 micro-kernels
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64.S.in      -D INC=0 -D PREFETCH=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64.S.in      -D INC=0 -D PREFETCH=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-prfm.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64-acc2.S.in -D INC=0 -D PREFETCH=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64-acc2.S.in -D INC=0 -D PREFETCH=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2-prfm.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64-acc4.S.in -D INC=0 -D PREFETCH=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64-acc4.S.in -D INC=0 -D PREFETCH=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4-prfm.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld64.S.in      -D INC=0               -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-asm-aarch64-neonfma-ld64.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld64.S.in      -D INC=0               -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-asm-aarch64-neonfma-ld64.S &

### LD128 micro-kernels
tools/xngen src/f32-gemm/1x8-aarch64-neon-ld128-acc2.S.in             -D INC=0 -D PREFETCH=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2.S &
tools/xngen src/f32-gemm/1x8-aarch64-neon-ld128-acc2.S.in             -D INC=0 -D PREFETCH=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2-prfm.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128.S.in      -D GOI=0 -D INC=0 -D PREFETCH=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128.S.in      -D GOI=0 -D INC=0 -D PREFETCH=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-prfm.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128-acc2.S.in          -D INC=0 -D PREFETCH=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128-acc2.S.in          -D INC=0 -D PREFETCH=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2-prfm.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128-acc4.S.in          -D INC=0 -D PREFETCH=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128-acc4.S.in          -D INC=0 -D PREFETCH=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4-prfm.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld128.S.in      -D GOI=0 -D INC=0               -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-asm-aarch64-neonfma-ld128.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld128.S.in               -D INC=0               -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-asm-aarch64-neonfma-ld128.S &

### MRx1 micro-kernels
tools/xngen src/f32-gemm/4x1-aarch64-neonfma-ld64.S.in      -D INC=0                -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x1-minmax-asm-aarch64-neonfma-ld64.S &
tools/xngen src/f32-gemm/4x1-aarch64-neonfma-ld128.S.in     -D INC=0                -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x1-minmax-asm-aarch64-neonfma-ld128.S &
### MRx2 micro-kernels
tools/xngen src/f32-gemm/4x2-aarch64-neonfma-ld64.S.in      -D INC=0                -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x2-minmax-asm-aarch64-neonfma-ld64.S &
tools/xngen src/f32-gemm/4x2-aarch64-neonfma-ld128.S.in     -D INC=0                -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x2-minmax-asm-aarch64-neonfma-ld128.S &

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/f32-qc4w-gemm-minmax.yaml --output test/f32-qc4w-gemm-minmax.cc &

wait
