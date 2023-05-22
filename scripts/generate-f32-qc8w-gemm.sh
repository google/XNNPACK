#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x4-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x4-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x4-minmax-scalar.c &

tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-2x4-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-2x4-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-2x4-minmax-scalar.c &

tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x2-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x2-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x2-minmax-scalar.c &

tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x4-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x4-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x4-minmax-scalar.c &

### WAsm-specific micro-kernels
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x4-relu-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-2x4-relu-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x2-relu-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x4-relu-wasm.c &

tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x4-minmax-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-2x4-minmax-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x2-minmax-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x4-minmax-wasm.c &

############################### AArch64 assembly ##############################
### LD64 micro-kernels
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64.S.in      -D INC=0 -D PREFETCH=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64.S.in      -D INC=0 -D PREFETCH=1 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-prfm.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64-acc2.S.in -D INC=0 -D PREFETCH=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64-acc2.S.in -D INC=0 -D PREFETCH=1 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc2-prfm.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64-acc4.S.in -D INC=0 -D PREFETCH=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64-acc4.S.in -D INC=0 -D PREFETCH=1 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld64-acc4-prfm.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld64.S.in      -D INC=0               -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x8-minmax-asm-aarch64-neonfma-ld64.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld64.S.in      -D INC=0               -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-6x8-minmax-asm-aarch64-neonfma-ld64.S &

### LD128 micro-kernels
tools/xngen src/f32-gemm/1x8-aarch64-neon-ld128-acc2.S.in    -D INC=0 -D PREFETCH=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2.S &
tools/xngen src/f32-gemm/1x8-aarch64-neon-ld128-acc2.S.in    -D INC=0 -D PREFETCH=1 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neon-ld128-acc2-prfm.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128.S.in      -D INC=0 -D PREFETCH=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128.S.in      -D INC=0 -D PREFETCH=1 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-prfm.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128-acc2.S.in -D INC=0 -D PREFETCH=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128-acc2.S.in -D INC=0 -D PREFETCH=1 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc2-prfm.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128-acc4.S.in -D INC=0 -D PREFETCH=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld128-acc4.S.in -D INC=0 -D PREFETCH=1 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-asm-aarch64-neonfma-ld128-acc4-prfm.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld128.S.in      -D INC=0               -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x8-minmax-asm-aarch64-neonfma-ld128.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld128.S.in      -D INC=0               -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-6x8-minmax-asm-aarch64-neonfma-ld128.S &

### MRx2 micro-kernels
tools/xngen src/f32-gemm/4x2-aarch64-neonfma-ld64.S.in      -D INC=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x2-minmax-asm-aarch64-neonfma-ld64.S &

################################### ARM NEON ##################################
### LD64 micro-kernels
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=4 -D NR=8 -D FMA=1 -D INC=0 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x8-minmax-aarch64-neonfma-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=6 -D NR=8 -D FMA=1 -D INC=0 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-6x8-minmax-aarch64-neonfma-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=1 -D NR=8 -D FMA=0 -D INC=0 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=4 -D NR=8 -D FMA=0 -D INC=0 -D DUP=0 -D DATATYPE=QC8  -o src/f32-gemm/gen/f32-qc8w-gemm-4x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=5 -D NR=8 -D FMA=0 -D INC=0 -D DUP=0 -D DATATYPE=QC8  -o src/f32-gemm/gen/f32-qc8w-gemm-5x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=5 -D NR=8 -D FMA=1 -D INC=0 -D DUP=0 -D DATATYPE=QC8  -o src/f32-gemm/gen/f32-qc8w-gemm-5x8-minmax-aarch64-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=6 -D NR=8 -D FMA=0 -D INC=0 -D DUP=0 -D DATATYPE=QC8  -o src/f32-gemm/gen/f32-qc8w-gemm-6x8-minmax-neon-lane-ld64.c &

### LD128 micro-kernels
tools/xngen src/f32-gemm/neon-ld128.c.in -D MR=1 -D NR=8  -D INC=0 -D FMA=1 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-aarch64-neonfma-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in -D MR=4 -D NR=8  -D INC=0 -D FMA=1 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x8-minmax-aarch64-neonfma-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in -D MR=6 -D NR=8  -D INC=0 -D FMA=1 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-6x8-minmax-aarch64-neonfma-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in -D MR=1 -D NR=16 -D INC=0 -D FMA=1 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x16-minmax-aarch64-neonfma-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in -D MR=4 -D NR=16 -D INC=0 -D FMA=1 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x16-minmax-aarch64-neonfma-ld128.c &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld128.S.in      -D INC=0 -D DATATYPE=F32 -o src/f32-gemm/gen/f32-gemm-4x8-minmax-asm-aarch64-neonfma-ld128.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld128.S.in      -D INC=1 -D DATATYPE=F32 -o src/f32-gemm/gen/f32-gemminc-4x8-minmax-asm-aarch64-neonfma-ld128.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld128.S.in      -D INC=0 -D DATATYPE=F32 -o src/f32-gemm/gen/f32-gemm-6x8-minmax-asm-aarch64-neonfma-ld128.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld128.S.in      -D INC=1 -D DATATYPE=F32 -o src/f32-gemm/gen/f32-gemminc-6x8-minmax-asm-aarch64-neonfma-ld128.S &

### MRx2 micro-kernels
tools/xngen src/f32-gemm/MRx2-neon-ld64.c.in -D MR=4 -D NR=2  -D FMA=0 -D INC=0 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x2-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/MRx2-neon-ld64.c.in -D MR=4 -D NR=2  -D FMA=1 -D INC=0 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x2-minmax-aarch64-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/MRx2-neon-ld64.c.in -D MR=6 -D NR=2  -D FMA=0 -D INC=0 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-6x2-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/MRx2-neon-ld64.c.in -D MR=6 -D NR=2  -D FMA=1 -D INC=0 -D DUP=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-6x2-minmax-aarch64-neonfma-lane-ld64.c &
### DUP LD64 micro-kernels
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=1 -D NR=8 -D FMA=0 -D INC=0 -D DUP=1 -D DATATYPE=QC8  -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=1 -D NR=8 -D FMA=1 -D INC=0 -D DUP=1 -D DATATYPE=QC8  -o src/f32-gemm/gen/f32-qc8w-gemm-1x8-minmax-neonfma-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=4 -D NR=8 -D FMA=0 -D INC=0 -D DUP=1 -D DATATYPE=QC8  -o src/f32-gemm/gen/f32-qc8w-gemm-4x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=4 -D NR=8 -D FMA=1 -D INC=0 -D DUP=1 -D DATATYPE=QC8  -o src/f32-gemm/gen/f32-qc8w-gemm-4x8-minmax-neonfma-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=6 -D NR=8 -D FMA=0 -D INC=0 -D DUP=1 -D DATATYPE=QC8  -o src/f32-gemm/gen/f32-qc8w-gemm-6x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in -D MR=6 -D NR=8 -D FMA=1 -D INC=0 -D DUP=1 -D DATATYPE=QC8  -o src/f32-gemm/gen/f32-qc8w-gemm-6x8-minmax-neonfma-dup-ld64.c &
### SHUFFLE micro-kernels
tools/xngen src/f32-gemm/neon-shuffle.c.in -D MR=1 -D NR=8  -D FMA=1 -D INC=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-1x8s4-minmax-neonfma.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in -D MR=4 -D NR=8  -D FMA=1 -D INC=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-4x8s4-minmax-neonfma.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in -D MR=6 -D NR=8  -D FMA=1 -D INC=0 -D DATATYPE=QC8 -o src/f32-gemm/gen/f32-qc8w-gemm-6x8s4-minmax-neonfma.c &

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/f32-qc8w-gemm.yaml        --output test/f32-qc8w-gemm.cc        &
tools/generate-gemm-test.py --spec test/f32-qc8w-gemm-relu.yaml   --output test/f32-qc8w-gemm-relu.cc   &
tools/generate-gemm-test.py --spec test/f32-qc8w-gemm-minmax.yaml --output test/f32-qc8w-gemm-minmax.cc &

wait
