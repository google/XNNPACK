#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
#tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x4-scalar.c &
#tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x4-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x4-minmax-scalar.c &

#tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-2x4-scalar.c &
#tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-2x4-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-2x4-minmax-scalar.c &

#tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x2-scalar.c &
#tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x2-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x2-minmax-scalar.c &

#tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x4-scalar.c &
#tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x4-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x4-minmax-scalar.c &

### WAsm-specific micro-kernels
#tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x4-relu-wasm.c &
#tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-2x4-relu-wasm.c &
#tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x2-relu-wasm.c &
#tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x4-relu-wasm.c &

tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x4-minmax-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-2x4-minmax-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x2-minmax-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x4-minmax-wasm.c &

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

################################### ARM NEON ##################################
### LD64 micro-kernels
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=1 -D NR=8 -D FMA=1 -D INC=0 -D DUP=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-aarch64-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=4 -D NR=8 -D FMA=1 -D INC=0 -D DUP=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-aarch64-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=5 -D NR=8 -D FMA=1 -D INC=0 -D DUP=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-5x8-minmax-aarch64-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=6 -D NR=8 -D FMA=1 -D INC=0 -D DUP=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-aarch64-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=1 -D NR=8 -D FMA=0 -D INC=0 -D DUP=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=4 -D NR=8 -D FMA=0 -D INC=0 -D DUP=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=5 -D NR=8 -D FMA=0 -D INC=0 -D DUP=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-5x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=6 -D NR=8 -D FMA=0 -D INC=0 -D DUP=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-neon-lane-ld64.c &

### LD128 micro-kernels
tools/xngen src/f32-gemm/neon-ld128.c.in -D MR=1 -D NR=8  -D INC=0 -D FMA=1 -D DUP=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-aarch64-neonfma-lane-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in -D MR=4 -D NR=8  -D INC=0 -D FMA=1 -D DUP=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-aarch64-neonfma-lane-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in -D MR=6 -D NR=8  -D INC=0 -D FMA=1 -D DUP=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-aarch64-neonfma-lane-ld128.c &

### DUP LD64 micro-kernels
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=1 -D NR=8 -D FMA=0 -D INC=0 -D DUP=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=4 -D NR=8 -D FMA=0 -D INC=0 -D DUP=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=6 -D NR=8 -D FMA=0 -D INC=0 -D DUP=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=1 -D NR=8 -D FMA=1 -D INC=0 -D DUP=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-neonfma-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=4 -D NR=8 -D FMA=1 -D INC=0 -D DUP=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-neonfma-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=6 -D NR=8 -D FMA=1 -D INC=0 -D DUP=1 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-neonfma-dup-ld64.c &

################################### x86 SSE ###################################
### LOAD4+DUPLICATE micro-kernels
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=1 -D NR=8 -D INC=0 -D SSE=2 -D AVX=0 -D FMA=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-sse2-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=3 -D NR=8 -D INC=0 -D SSE=2 -D AVX=0 -D FMA=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-3x8-minmax-sse2-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=4 -D NR=8 -D INC=0 -D SSE=2 -D AVX=0 -D FMA=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-sse2-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=5 -D NR=8 -D INC=0 -D SSE=2 -D AVX=0 -D FMA=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-5x8-minmax-sse2-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=6 -D NR=8 -D INC=0 -D SSE=2 -D AVX=0 -D FMA=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-sse2-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=1 -D NR=8 -D INC=0 -D SSE=4 -D AVX=0 -D FMA=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x8-minmax-sse41-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=3 -D NR=8 -D INC=0 -D SSE=4 -D AVX=0 -D FMA=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-3x8-minmax-sse41-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=4 -D NR=8 -D INC=0 -D SSE=4 -D AVX=0 -D FMA=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x8-minmax-sse41-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=5 -D NR=8 -D INC=0 -D SSE=4 -D AVX=0 -D FMA=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-5x8-minmax-sse41-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=6 -D NR=8 -D INC=0 -D SSE=4 -D AVX=0 -D FMA=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x8-minmax-sse41-dup.c &

################################### x86 AVX ###################################
### AVX BROADCAST micro-kernels
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=1 -D NR=16 -D AVX=1 -D FMA=0 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x16-minmax-avx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=2 -D NR=16 -D AVX=1 -D FMA=0 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-2x16-minmax-avx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=3 -D NR=16 -D AVX=1 -D FMA=0 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-3x16-minmax-avx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=4 -D NR=16 -D AVX=1 -D FMA=0 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x16-minmax-avx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D AVX=1 -D FMA=0 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-5x16-minmax-avx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=6 -D NR=16 -D AVX=1 -D FMA=0 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x16-minmax-avx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=7 -D NR=16 -D AVX=1 -D FMA=0 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-7x16-minmax-avx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=8 -D NR=16 -D AVX=1 -D FMA=0 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-8x16-minmax-avx-broadcast.c &

### AVX FMA3+BROADCAST micro-kernels
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=1 -D NR=16 -D AVX=1 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=2 -D NR=16 -D AVX=1 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-2x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=3 -D NR=16 -D AVX=1 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-3x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=4 -D NR=16 -D AVX=1 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D AVX=1 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-5x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=6 -D NR=16 -D AVX=1 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=7 -D NR=16 -D AVX=1 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-7x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=8 -D NR=16 -D AVX=1 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-8x16-minmax-fma3-broadcast.c &

### AVX2 FMA3+BROADCAST micro-kernels
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=1 -D NR=16 -D AVX=2 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-1x16-minmax-avx2-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=2 -D NR=16 -D AVX=2 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-2x16-minmax-avx2-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=3 -D NR=16 -D AVX=2 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-3x16-minmax-avx2-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=4 -D NR=16 -D AVX=2 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-4x16-minmax-avx2-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D AVX=2 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-5x16-minmax-avx2-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=6 -D NR=16 -D AVX=2 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-6x16-minmax-avx2-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=7 -D NR=16 -D AVX=2 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-7x16-minmax-avx2-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx-broadcast.c.in -D MR=8 -D NR=16 -D AVX=2 -D FMA=3 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc4w-gemm/gen/f32-qc4w-gemm-8x16-minmax-avx2-broadcast.c &

################################# x86 AVX-512 #################################
### AVX512SKX+BROADCAST micro-kernels
tools/xngen src/f32-qc4w-gemm/avx512-broadcast.c.in -D MR=1 -D NR=32 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc8w-gemm/gen/f32-qc4w-gemm-1x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx512-broadcast.c.in -D MR=2 -D NR=32 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc8w-gemm/gen/f32-qc4w-gemm-2x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx512-broadcast.c.in -D MR=3 -D NR=32 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc8w-gemm/gen/f32-qc4w-gemm-3x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx512-broadcast.c.in -D MR=4 -D NR=32 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc8w-gemm/gen/f32-qc4w-gemm-4x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx512-broadcast.c.in -D MR=5 -D NR=32 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc8w-gemm/gen/f32-qc4w-gemm-5x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx512-broadcast.c.in -D MR=6 -D NR=32 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc8w-gemm/gen/f32-qc4w-gemm-6x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx512-broadcast.c.in -D MR=7 -D NR=32 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc8w-gemm/gen/f32-qc4w-gemm-7x32-minmax-avx512skx-broadcast.c &
tools/xngen src/f32-qc4w-gemm/avx512-broadcast.c.in -D MR=8 -D NR=32 -D INC=0 -D DATATYPE=QC4 -o src/f32-qc8w-gemm/gen/f32-qc4w-gemm-8x32-minmax-avx512skx-broadcast.c &

wait
