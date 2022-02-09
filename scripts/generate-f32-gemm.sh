#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR  -o src/f32-gemm/gen/1x4-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=RELU    -o src/f32-gemm/gen/1x4-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX  -o src/f32-gemm/gen/1x4-minmax-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D INC=1 -D ACTIVATION=MINMAX  -o src/f32-gemm/gen-inc/1x4inc-minmax-scalar.c &

tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR -o src/f32-gemm/gen/2x4-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=RELU   -o src/f32-gemm/gen/2x4-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/2x4-minmax-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/2x4inc-minmax-scalar.c &

tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR -o src/f32-gemm/gen/4x2-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D INC=0 -D ACTIVATION=RELU   -o src/f32-gemm/gen/4x2-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x2-minmax-scalar.c &

tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=LINEAR -o src/f32-gemm/gen/4x4-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=RELU   -o src/f32-gemm/gen/4x4-relu-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x4-minmax-scalar.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/4x4inc-minmax-scalar.c &

### WAsm-specific micro-kernels
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=LINEAR -o src/f32-gemm/gen/1x4-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -o src/f32-gemm/gen/1x4-relu-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/1x4-minmax-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/1x4inc-minmax-wasm.c &

tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=LINEAR -o src/f32-gemm/gen/2x4-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -o src/f32-gemm/gen/2x4-relu-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/2x4-minmax-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/2x4inc-minmax-wasm.c &

tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=1 -D INC=0 -D ACTIVATION=LINEAR -o src/f32-gemm/gen/4x2-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -o src/f32-gemm/gen/4x2-relu-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x2-minmax-wasm.c &

tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=LINEAR -o src/f32-gemm/gen/4x4-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=RELU   -o src/f32-gemm/gen/4x4-relu-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x4-minmax-wasm.c &
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/4x4inc-minmax-wasm.c &

############################### AArch64 assembly ##############################

### LD64 micro-kernels
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld64.S.in        -D INC=0 -o src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-ld64.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld64.S.in        -D INC=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-ld64.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld64.S.in        -D INC=0 -o src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-ld64.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld64.S.in        -D INC=1 -o src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-ld64.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64.S.in        -D INC=0 -o src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-ld64.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-ld64.S.in        -D INC=1 -o src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-ld64.S &

### LD128 micro-kernels
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld128.S.in       -D INC=0 -o src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-ld128.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld128.S.in       -D INC=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-ld128.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld128.S.in       -D INC=0 -o src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-ld128.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld128.S.in       -D INC=1 -o src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-ld128.S &

### Cortex A53 micro-kernels
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-cortex-a53.S.in  -D INC=0 -D PREFETCH=0 -o src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-cortex-a53.S.in  -D INC=0 -D PREFETCH=1 -o src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-prfm-cortex-a53.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-cortex-a53.S.in  -D INC=1 -D PREFETCH=0 -o src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a53.S.in  -D INC=0 -D PREFETCH=0 -o src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a53.S.in  -D INC=0 -D PREFETCH=1 -o src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-prfm-cortex-a53.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a53.S.in  -D INC=1 -D PREFETCH=0 -o src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-gemm/4x12-aarch64-neonfma-cortex-a53.S.in -D INC=0 -o src/f32-gemm/gen/4x12-minmax-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-gemm/4x12-aarch64-neonfma-cortex-a53.S.in -D INC=1 -o src/f32-gemm/gen-inc/4x12inc-minmax-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-gemm/1x12-aarch64-neonfma-cortex-a53.S.in -D INC=0 -o src/f32-gemm/gen/1x12-minmax-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-gemm/1x12-aarch64-neonfma-cortex-a53.S.in -D INC=1 -o src/f32-gemm/gen-inc/1x12inc-minmax-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-cortex-a53.S.in  -D INC=0 -o src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-cortex-a53.S.in  -D INC=1 -o src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-cortex-a53.S &

### Cortex A55 micro-kernels
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-cortex-a55.S.in  -D INC=0 -o src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a55.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-cortex-a55.S.in  -D INC=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-cortex-a55.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a55.S.in  -D INC=0 -o src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a55.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a55.S.in  -D INC=1 -o src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a55.S &

### Cortex A75 micro-kernels
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=0 -o src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-cortex-a75.S.in  -D INC=1 -D PREFETCH=0 -o src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=1 -o src/f32-gemm/gen/4x8-minmax-aarch64-neonfma-prfm-cortex-a75.S &
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-cortex-a75.S.in  -D INC=1 -D PREFETCH=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-aarch64-neonfma-prfm-cortex-a75.S &
tools/xngen src/f32-gemm/5x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=0 -o src/f32-gemm/gen/5x8-minmax-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-gemm/5x8-aarch64-neonfma-cortex-a75.S.in  -D INC=1 -D PREFETCH=0 -o src/f32-gemm/gen-inc/5x8inc-minmax-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-gemm/5x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=1 -o src/f32-gemm/gen/5x8-minmax-aarch64-neonfma-prfm-cortex-a75.S &
tools/xngen src/f32-gemm/5x8-aarch64-neonfma-cortex-a75.S.in  -D INC=1 -D PREFETCH=1 -o src/f32-gemm/gen-inc/5x8inc-minmax-aarch64-neonfma-prfm-cortex-a75.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=0 -o src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a75.S.in  -D INC=1 -D PREFETCH=0 -o src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=1 -o src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-prfm-cortex-a75.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a75.S.in  -D INC=1 -D PREFETCH=1 -o src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-prfm-cortex-a75.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=0 -o src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-cortex-a75.S.in  -D INC=1 -D PREFETCH=0 -o src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=1 -o src/f32-gemm/gen/1x8-minmax-aarch64-neonfma-prfm-cortex-a75.S &
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-cortex-a75.S.in  -D INC=1 -D PREFETCH=1 -o src/f32-gemm/gen-inc/1x8inc-minmax-aarch64-neonfma-prfm-cortex-a75.S &

### Cortex A73 micro-kernels
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a73.S.in  -D INC=0 -o src/f32-gemm/gen/6x8-minmax-aarch64-neonfma-cortex-a73.S &
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a73.S.in  -D INC=1 -o src/f32-gemm/gen-inc/6x8inc-minmax-aarch64-neonfma-cortex-a73.S &

############################### AArch32 assembly ##############################
tools/xngen src/f32-gemm/4x8-aarch32-neon-cortex-a53.S.in       -D INC=0 -D PREFETCH=0 -o src/f32-gemm/gen/4x8-minmax-aarch32-neon-cortex-a53.S &
tools/xngen src/f32-gemm/4x8-aarch32-neon-cortex-a53.S.in       -D INC=0 -D PREFETCH=1 -o src/f32-gemm/gen/4x8-minmax-aarch32-neon-prfm-cortex-a53.S &
tools/xngen src/f32-gemm/4x8-aarch32-neon-cortex-a75.S.in       -D INC=0 -D PREFETCH=0 -o src/f32-gemm/gen/4x8-minmax-aarch32-neon-cortex-a75.S &
tools/xngen src/f32-gemm/4x8-aarch32-neon-cortex-a75.S.in       -D INC=0 -D PREFETCH=1 -o src/f32-gemm/gen/4x8-minmax-aarch32-neon-prfm-cortex-a75.S &
tools/xngen src/f32-gemm/4x8-minmax-aarch32-neon-cortex-a7.S.in -D INC=0 -D PREFETCH=1 -o src/f32-gemm/gen/4x8-minmax-aarch32-neon-cortex-a7.S &
tools/xngen src/f32-gemm/4x8-minmax-aarch32-neon-ld64.S.in      -D INC=0 -D PREFETCH=0 -o src/f32-gemm/gen/4x8-minmax-aarch32-neon-ld64.S &

################################### ARM NEON ##################################
### LD64 micro-kernels
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=1 -D NR=8  -D FMA=0 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/1x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=1 -D NR=8  -D FMA=0 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/1x8inc-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=1 -D NR=8  -D FMA=1 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/1x8-minmax-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=1 -D NR=8  -D FMA=1 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/1x8inc-minmax-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=8  -D FMA=0 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/4x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=8  -D FMA=0 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/4x8inc-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=8  -D FMA=1 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/4x8-minmax-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=8  -D FMA=1 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=5 -D NR=8  -D FMA=0 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/5x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=5 -D NR=8  -D FMA=0 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/5x8inc-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=5 -D NR=8  -D FMA=1 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/5x8-minmax-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=5 -D NR=8  -D FMA=1 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/5x8inc-minmax-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=6 -D NR=8  -D FMA=0 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/6x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=6 -D NR=8  -D FMA=0 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/6x8inc-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=6 -D NR=8  -D FMA=1 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/6x8-minmax-neonfma-lane-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=6 -D NR=8  -D FMA=1 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-lane-ld64.c &
### LD128 micro-kernels
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=0 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/4x8-minmax-neon-lane-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=0 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/4x8inc-minmax-neon-lane-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=1 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/4x8-minmax-neonfma-lane-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=1 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-lane-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=6 -D NR=8  -D FMA=0 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/6x8-minmax-neon-lane-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=6 -D NR=8  -D FMA=0 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/6x8inc-minmax-neon-lane-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=6 -D NR=8  -D FMA=1 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/6x8-minmax-neonfma-lane-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=6 -D NR=8  -D FMA=1 -D INC=1 -D DUP=0 -o src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-lane-ld128.c &
### MRx2 micro-kernels
tools/xngen src/f32-gemm/MRx2-neon-ld64.c.in -D MR=4 -D NR=2  -D FMA=0 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/4x2-minmax-neon-lane-ld64.c &
tools/xngen src/f32-gemm/MRx2-neon-ld64.c.in -D MR=4 -D NR=2  -D FMA=1 -D INC=0 -D DUP=0 -o src/f32-gemm/gen/4x2-minmax-neonfma-lane-ld64.c &
### DUP LD64 micro-kernels
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=1 -D NR=8  -D FMA=0 -D INC=0 -D DUP=1 -o src/f32-gemm/gen/1x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=1 -D NR=8  -D FMA=0 -D INC=1 -D DUP=1 -o src/f32-gemm/gen-inc/1x8inc-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=1 -D NR=8  -D FMA=1 -D INC=0 -D DUP=1 -o src/f32-gemm/gen/1x8-minmax-neonfma-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=1 -D NR=8  -D FMA=1 -D INC=1 -D DUP=1 -o src/f32-gemm/gen-inc/1x8inc-minmax-neonfma-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=8  -D FMA=0 -D INC=0 -D DUP=1 -o src/f32-gemm/gen/4x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=8  -D FMA=0 -D INC=1 -D DUP=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=8  -D FMA=1 -D INC=0 -D DUP=1 -o src/f32-gemm/gen/4x8-minmax-neonfma-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=8  -D FMA=1 -D INC=1 -D DUP=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=6 -D NR=8  -D FMA=0 -D INC=0 -D DUP=1 -o src/f32-gemm/gen/6x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=6 -D NR=8  -D FMA=0 -D INC=1 -D DUP=1 -o src/f32-gemm/gen-inc/6x8inc-minmax-neon-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=6 -D NR=8  -D FMA=1 -D INC=0 -D DUP=1 -o src/f32-gemm/gen/6x8-minmax-neonfma-dup-ld64.c &
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=6 -D NR=8  -D FMA=1 -D INC=1 -D DUP=1 -o src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-dup-ld64.c &
### DUP LD128 midupkernels
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=0 -D INC=0 -D DUP=1 -o src/f32-gemm/gen/4x8-minmax-neon-dup-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=0 -D INC=1 -D DUP=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-neon-dup-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=1 -D INC=0 -D DUP=1 -o src/f32-gemm/gen/4x8-minmax-neonfma-dup-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=1 -D INC=1 -D DUP=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-neonfma-dup-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=6 -D NR=8  -D FMA=0 -D INC=0 -D DUP=1 -o src/f32-gemm/gen/6x8-minmax-neon-dup-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=6 -D NR=8  -D FMA=0 -D INC=1 -D DUP=1 -o src/f32-gemm/gen-inc/6x8inc-minmax-neon-dup-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=6 -D NR=8  -D FMA=1 -D INC=0 -D DUP=1 -o src/f32-gemm/gen/6x8-minmax-neonfma-dup-ld128.c &
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=6 -D NR=8  -D FMA=1 -D INC=1 -D DUP=1 -o src/f32-gemm/gen-inc/6x8inc-minmax-neonfma-dup-ld128.c &
### LOAD4+PERMUTE micro-kernels
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=1 -D NR=8  -D FMA=0 -D INC=0 -o src/f32-gemm/gen/1x8s4-minmax-neon.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=1 -D NR=8  -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/1x8s4inc-minmax-neon.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=1 -D NR=8  -D FMA=1 -D INC=0 -o src/f32-gemm/gen/1x8s4-minmax-neonfma.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=1 -D NR=8  -D FMA=1 -D INC=1 -o src/f32-gemm/gen-inc/1x8s4inc-minmax-neonfma.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=4 -D NR=8  -D FMA=0 -D INC=0 -o src/f32-gemm/gen/4x8s4-minmax-neon.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=4 -D NR=8  -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/4x8s4inc-minmax-neon.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=4 -D NR=8  -D FMA=1 -D INC=0 -o src/f32-gemm/gen/4x8s4-minmax-neonfma.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=4 -D NR=8  -D FMA=1 -D INC=1 -o src/f32-gemm/gen-inc/4x8s4inc-minmax-neonfma.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=6 -D NR=8  -D FMA=0 -D INC=0 -o src/f32-gemm/gen/6x8s4-minmax-neon.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=6 -D NR=8  -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/6x8s4inc-minmax-neon.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=6 -D NR=8  -D FMA=1 -D INC=0 -o src/f32-gemm/gen/6x8s4-minmax-neonfma.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=6 -D NR=8  -D FMA=1 -D INC=1 -o src/f32-gemm/gen-inc/6x8s4inc-minmax-neonfma.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=8 -D NR=8  -D FMA=0 -D INC=0 -o src/f32-gemm/gen/8x8s4-minmax-neon.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=8 -D NR=8  -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/8x8s4inc-minmax-neon.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=8 -D NR=8  -D FMA=1 -D INC=0 -o src/f32-gemm/gen/8x8s4-minmax-neonfma.c &
tools/xngen src/f32-gemm/neon-shuffle.c.in   -D MR=8 -D NR=8  -D FMA=1 -D INC=1 -o src/f32-gemm/gen-inc/8x8s4inc-minmax-neonfma.c &

################################## WAsm SIMD ##################################
### LOAD1+BROADCAST micro-kernels
tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/1x8-minmax-wasmsimd-arm-loadsplat.c &
tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/1x8inc-minmax-wasmsimd-arm-loadsplat.c &

tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/3x8-minmax-wasmsimd-arm-loadsplat.c &
tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/3x8inc-minmax-wasmsimd-arm-loadsplat.c &

tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x8-minmax-wasmsimd-arm-loadsplat.c &
tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/4x8inc-minmax-wasmsimd-arm-loadsplat.c &

tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/5x8-minmax-wasmsimd-arm-loadsplat.c &
tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/5x8inc-minmax-wasmsimd-arm-loadsplat.c &

tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/6x8-minmax-wasmsimd-arm-loadsplat.c &
tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/6x8inc-minmax-wasmsimd-arm-loadsplat.c &

tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/1x8-minmax-wasmsimd-x86-loadsplat.c &
tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/1x8inc-minmax-wasmsimd-x86-loadsplat.c &

tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/3x8-minmax-wasmsimd-x86-loadsplat.c &
tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/3x8inc-minmax-wasmsimd-x86-loadsplat.c &

tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x8-minmax-wasmsimd-x86-loadsplat.c &
tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/4x8inc-minmax-wasmsimd-x86-loadsplat.c &

tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/5x8-minmax-wasmsimd-x86-loadsplat.c &
tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/5x8inc-minmax-wasmsimd-x86-loadsplat.c &

tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/6x8-minmax-wasmsimd-x86-loadsplat.c &
tools/xngen src/f32-gemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/6x8inc-minmax-wasmsimd-x86-loadsplat.c &
### LOAD4+DUPLICATE micro-kernels
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/1x8-minmax-wasmsimd-arm-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/1x8inc-minmax-wasmsimd-arm-splat.c &

tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/3x8-minmax-wasmsimd-arm-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/3x8inc-minmax-wasmsimd-arm-splat.c &

tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x8-minmax-wasmsimd-arm-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/4x8inc-minmax-wasmsimd-arm-splat.c &

tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/5x8-minmax-wasmsimd-arm-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/5x8inc-minmax-wasmsimd-arm-splat.c &

tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/6x8-minmax-wasmsimd-arm-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/6x8inc-minmax-wasmsimd-arm-splat.c &

tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=RELU   -o src/f32-gemm/gen/1x8-relu-wasmsimd-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=RELU   -o src/f32-gemm/gen/4x8-relu-wasmsimd-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=RELU   -o src/f32-gemm/gen/5x8-relu-wasmsimd-splat.c &

tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=LINEAR -o src/f32-gemm/gen/1x8-wasmsimd-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=LINEAR -o src/f32-gemm/gen/4x8-wasmsimd-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=LINEAR -o src/f32-gemm/gen/5x8-wasmsimd-splat.c &

tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/1x8-minmax-wasmsimd-x86-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/1x8inc-minmax-wasmsimd-x86-splat.c &

tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/3x8-minmax-wasmsimd-x86-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/3x8inc-minmax-wasmsimd-x86-splat.c &

tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x8-minmax-wasmsimd-x86-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/4x8inc-minmax-wasmsimd-x86-splat.c &

tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/5x8-minmax-wasmsimd-x86-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/5x8inc-minmax-wasmsimd-x86-splat.c &

tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/6x8-minmax-wasmsimd-x86-splat.c &
tools/xngen src/f32-gemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/6x8inc-minmax-wasmsimd-x86-splat.c &

### LOAD4+PERMUTE micro-kernels
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/1x8s4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/1x8s4inc-minmax-wasmsimd-arm.c &

tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/3x8s4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/3x8s4inc-minmax-wasmsimd-arm.c &

tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x8s4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/4x8s4inc-minmax-wasmsimd-arm.c &

tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/5x8s4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/5x8s4inc-minmax-wasmsimd-arm.c &

tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/6x8s4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/6x8s4inc-minmax-wasmsimd-arm.c &

tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D X86=0 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/1x8s4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D X86=0 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/1x8s4inc-minmax-wasmsimd-arm.c &

tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/1x8s4-minmax-wasmsimd-x86.c &
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/1x8s4inc-minmax-wasmsimd-x86.c &

tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/3x8s4-minmax-wasmsimd-x86.c &
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/3x8s4inc-minmax-wasmsimd-x86.c &

tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x8s4-minmax-wasmsimd-x86.c &
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/4x8s4inc-minmax-wasmsimd-x86.c &

tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/5x8s4-minmax-wasmsimd-x86.c &
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/5x8s4inc-minmax-wasmsimd-x86.c &

tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D X86=1 -D INC=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/6x8s4-minmax-wasmsimd-x86.c &
tools/xngen src/f32-gemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D X86=1 -D INC=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen-inc/6x8s4inc-minmax-wasmsimd-x86.c &

### MRx2 micro-kernels
tools/xngen src/f32-gemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D X86=0 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x2c4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-gemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D X86=1 -D ACTIVATION=MINMAX -o src/f32-gemm/gen/4x2c4-minmax-wasmsimd-x86.c &
tools/xngen src/f32-gemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D X86=0 -D ACTIVATION=RELU   -o src/f32-gemm/gen/4x2c4-relu-wasmsimd.c &
tools/xngen src/f32-gemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D X86=0 -D ACTIVATION=LINEAR -o src/f32-gemm/gen/4x2c4-wasmsimd.c &

################################### x86 SSE ###################################
### LOAD1+BROADCAST micro-kernels
tools/xngen src/f32-gemm/sse-load1.c.in -D MR=1 -D NR=8 -D INC=0 -o src/f32-gemm/gen/1x8-minmax-sse-load1.c &
tools/xngen src/f32-gemm/sse-load1.c.in -D MR=1 -D NR=8 -D INC=1 -o src/f32-gemm/gen-inc/1x8inc-minmax-sse-load1.c &

tools/xngen src/f32-gemm/sse-load1.c.in -D MR=3 -D NR=8 -D INC=0 -o src/f32-gemm/gen/3x8-minmax-sse-load1.c &
tools/xngen src/f32-gemm/sse-load1.c.in -D MR=3 -D NR=8 -D INC=1 -o src/f32-gemm/gen-inc/3x8inc-minmax-sse-load1.c &

tools/xngen src/f32-gemm/sse-load1.c.in -D MR=4 -D NR=8 -D INC=0 -o src/f32-gemm/gen/4x8-minmax-sse-load1.c &
tools/xngen src/f32-gemm/sse-load1.c.in -D MR=4 -D NR=8 -D INC=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-sse-load1.c &

tools/xngen src/f32-gemm/sse-load1.c.in -D MR=5 -D NR=8 -D INC=0 -o src/f32-gemm/gen/5x8-minmax-sse-load1.c &
tools/xngen src/f32-gemm/sse-load1.c.in -D MR=5 -D NR=8 -D INC=1 -o src/f32-gemm/gen-inc/5x8inc-minmax-sse-load1.c &

### LOAD4+DUPLICATE micro-kernels
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=1 -D NR=8 -D INC=0 -D SSE=1 -o src/f32-gemm/gen/1x8-minmax-sse-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=1 -D NR=8 -D INC=1 -D SSE=1 -o src/f32-gemm/gen-inc/1x8inc-minmax-sse-dup.c &

tools/xngen src/f32-gemm/sse-dup.c.in -D MR=3 -D NR=8 -D INC=0 -D SSE=1 -o src/f32-gemm/gen/3x8-minmax-sse-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=3 -D NR=8 -D INC=1 -D SSE=1 -o src/f32-gemm/gen-inc/3x8inc-minmax-sse-dup.c &

tools/xngen src/f32-gemm/sse-dup.c.in -D MR=4 -D NR=8 -D INC=0 -D SSE=1 -o src/f32-gemm/gen/4x8-minmax-sse-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=4 -D NR=8 -D INC=1 -D SSE=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-sse-dup.c &

tools/xngen src/f32-gemm/sse-dup.c.in -D MR=5 -D NR=8 -D INC=0 -D SSE=1 -o src/f32-gemm/gen/5x8-minmax-sse-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=5 -D NR=8 -D INC=1 -D SSE=1 -o src/f32-gemm/gen-inc/5x8inc-minmax-sse-dup.c &

tools/xngen src/f32-gemm/sse-dup.c.in -D MR=1 -D NR=8 -D INC=0 -D SSE=2 -o src/f32-gemm/gen/1x8-minmax-sse2-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=1 -D NR=8 -D INC=1 -D SSE=2 -o src/f32-gemm/gen-inc/1x8inc-minmax-sse2-dup.c &

tools/xngen src/f32-gemm/sse-dup.c.in -D MR=3 -D NR=8 -D INC=0 -D SSE=2 -o src/f32-gemm/gen/3x8-minmax-sse2-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=3 -D NR=8 -D INC=1 -D SSE=2 -o src/f32-gemm/gen-inc/3x8inc-minmax-sse2-dup.c &

tools/xngen src/f32-gemm/sse-dup.c.in -D MR=4 -D NR=8 -D INC=0 -D SSE=2 -o src/f32-gemm/gen/4x8-minmax-sse2-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=4 -D NR=8 -D INC=1 -D SSE=2 -o src/f32-gemm/gen-inc/4x8inc-minmax-sse2-dup.c &

tools/xngen src/f32-gemm/sse-dup.c.in -D MR=5 -D NR=8 -D INC=0 -D SSE=2 -o src/f32-gemm/gen/5x8-minmax-sse2-dup.c &
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=5 -D NR=8 -D INC=1 -D SSE=2 -o src/f32-gemm/gen-inc/5x8inc-minmax-sse2-dup.c &

### LOAD4+PERMUTE micro-kernels
tools/xngen src/f32-gemm/sse-shuffle.c.in -D MR=1 -D NR=8 -D INC=0 -o src/f32-gemm/gen/1x8s4-minmax-sse.c &
tools/xngen src/f32-gemm/sse-shuffle.c.in -D MR=1 -D NR=8 -D INC=1 -o src/f32-gemm/gen-inc/1x8s4inc-minmax-sse.c &

tools/xngen src/f32-gemm/sse-shuffle.c.in -D MR=3 -D NR=8 -D INC=0 -o src/f32-gemm/gen/3x8s4-minmax-sse.c &
tools/xngen src/f32-gemm/sse-shuffle.c.in -D MR=3 -D NR=8 -D INC=1 -o src/f32-gemm/gen-inc/3x8s4inc-minmax-sse.c &

tools/xngen src/f32-gemm/sse-shuffle.c.in -D MR=4 -D NR=8 -D INC=0 -o src/f32-gemm/gen/4x8s4-minmax-sse.c &
tools/xngen src/f32-gemm/sse-shuffle.c.in -D MR=4 -D NR=8 -D INC=1 -o src/f32-gemm/gen-inc/4x8s4inc-minmax-sse.c &

tools/xngen src/f32-gemm/sse-shuffle.c.in -D MR=5 -D NR=8 -D INC=0 -o src/f32-gemm/gen/5x8s4-minmax-sse.c &
tools/xngen src/f32-gemm/sse-shuffle.c.in -D MR=5 -D NR=8 -D INC=1 -o src/f32-gemm/gen-inc/5x8s4inc-minmax-sse.c &

### MRx2 micro-kernels
tools/xngen src/f32-gemm/MRx2c4-sse.c.in -D MR=4 -D NR=2 -o src/f32-gemm/gen/4x2c4-minmax-sse.c &

################################### x86 AVX ###################################
### AVX+BROADCAST micro-kernels
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=1 -D NR=8 -D FMA=0 -D INC=0 -o src/f32-gemm/gen/1x8-minmax-avx-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=1 -D NR=8 -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/1x8inc-minmax-avx-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=4 -D NR=8 -D FMA=0 -D INC=0 -o src/f32-gemm/gen/4x8-minmax-avx-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=4 -D NR=8 -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-avx-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=5 -D NR=8 -D FMA=0 -D INC=0 -o src/f32-gemm/gen/5x8-minmax-avx-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=5 -D NR=8 -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/5x8inc-minmax-avx-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=6 -D NR=8 -D FMA=0 -D INC=0 -o src/f32-gemm/gen/6x8-minmax-avx-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=6 -D NR=8 -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/6x8inc-minmax-avx-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=7 -D NR=8 -D FMA=0 -D INC=0 -o src/f32-gemm/gen/7x8-minmax-avx-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=7 -D NR=8 -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/7x8inc-minmax-avx-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=1 -D NR=16 -D FMA=0 -D INC=0 -o src/f32-gemm/gen/1x16-minmax-avx-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=1 -D NR=16 -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/1x16inc-minmax-avx-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=3 -D NR=16 -D FMA=0 -D INC=0 -o src/f32-gemm/gen/3x16-minmax-avx-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=3 -D NR=16 -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/3x16inc-minmax-avx-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=4 -D NR=16 -D FMA=0 -D INC=0 -o src/f32-gemm/gen/4x16-minmax-avx-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=4 -D NR=16 -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/4x16inc-minmax-avx-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D FMA=0 -D INC=0 -o src/f32-gemm/gen/5x16-minmax-avx-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D FMA=0 -D INC=1 -o src/f32-gemm/gen-inc/5x16inc-minmax-avx-broadcast.c &
### FMA3+BROADCAST micro-kernels
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=1 -D NR=8 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/1x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=1 -D NR=8 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/1x8inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=4 -D NR=8 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/4x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=4 -D NR=8 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/4x8inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=5 -D NR=8 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/5x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=5 -D NR=8 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/5x8inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=6 -D NR=8 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/6x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=6 -D NR=8 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/6x8inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=7 -D NR=8 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/7x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=7 -D NR=8 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/7x8inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=8 -D NR=8 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/8x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=8 -D NR=8 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/8x8inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=1 -D NR=16 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/1x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=1 -D NR=16 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/1x16inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=3 -D NR=16 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/3x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=3 -D NR=16 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/3x16inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=4 -D NR=16 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/4x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=4 -D NR=16 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/4x16inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/5x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/5x16inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-shuffle4.c.in -D MR=1 -D NR=16 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/1x16s4-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-shuffle4.c.in -D MR=1 -D NR=16 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/1x16s4inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-shuffle4.c.in -D MR=3 -D NR=16 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/3x16s4-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-shuffle4.c.in -D MR=3 -D NR=16 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/3x16s4inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-shuffle4.c.in -D MR=4 -D NR=16 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/4x16s4-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-shuffle4.c.in -D MR=4 -D NR=16 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/4x16s4inc-minmax-fma3-broadcast.c &

tools/xngen src/f32-gemm/avx-shuffle4.c.in -D MR=5 -D NR=16 -D FMA=3 -D INC=0 -o src/f32-gemm/gen/5x16s4-minmax-fma3-broadcast.c &
tools/xngen src/f32-gemm/avx-shuffle4.c.in -D MR=5 -D NR=16 -D FMA=3 -D INC=1 -o src/f32-gemm/gen-inc/5x16s4inc-minmax-fma3-broadcast.c &

################################# x86 AVX-512 #################################
### AVX512F+BROADCAST micro-kernels
tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=1 -D NR=16 -D INC=0 -o src/f32-gemm/gen/1x16-minmax-avx512f-broadcast.c &
tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=1 -D NR=16 -D INC=1 -o src/f32-gemm/gen-inc/1x16inc-minmax-avx512f-broadcast.c &

tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=4 -D NR=16 -D INC=0 -o src/f32-gemm/gen/4x16-minmax-avx512f-broadcast.c &
tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=4 -D NR=16 -D INC=1 -o src/f32-gemm/gen-inc/4x16inc-minmax-avx512f-broadcast.c &

tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=5 -D NR=16 -D INC=0 -o src/f32-gemm/gen/5x16-minmax-avx512f-broadcast.c &
tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=5 -D NR=16 -D INC=1 -o src/f32-gemm/gen-inc/5x16inc-minmax-avx512f-broadcast.c &

tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=6 -D NR=16 -D INC=0 -o src/f32-gemm/gen/6x16-minmax-avx512f-broadcast.c &
tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=6 -D NR=16 -D INC=1 -o src/f32-gemm/gen-inc/6x16inc-minmax-avx512f-broadcast.c &

tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=7 -D NR=16 -D INC=0 -o src/f32-gemm/gen/7x16-minmax-avx512f-broadcast.c &
tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=7 -D NR=16 -D INC=1 -o src/f32-gemm/gen-inc/7x16inc-minmax-avx512f-broadcast.c &

tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=8 -D NR=16 -D INC=0 -o src/f32-gemm/gen/8x16-minmax-avx512f-broadcast.c &
tools/xngen src/f32-gemm/avx512-broadcast.c.in -D MR=8 -D NR=16 -D INC=1 -o src/f32-gemm/gen-inc/8x16inc-minmax-avx512f-broadcast.c &

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/f32-gemm.yaml --output test/f32-gemm.cc --output test/f32-gemm-2.cc &
tools/generate-gemm-test.py --spec test/f32-gemm-relu.yaml --output test/f32-gemm-relu.cc --output test/f32-gemm-relu-2.cc &
tools/generate-gemm-test.py --spec test/f32-gemm-minmax.yaml --output test/f32-gemm-minmax.cc --output test/f32-gemm-minmax-2.cc &
tools/generate-gemm-test.py --spec test/f32-gemminc-minmax.yaml --output test/f32-gemminc-minmax.cc --output test/f32-gemminc-minmax-2.cc &

wait
