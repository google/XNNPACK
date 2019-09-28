#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-gemm/scalar.c.in -D MR=1 -D NR=4 -D INC=0 -o src/f32-gemm/1x4-scalar.c
tools/xngen src/f32-gemm/scalar.c.in -D MR=2 -D NR=4 -D INC=0 -o src/f32-gemm/2x4-scalar.c
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=2 -D INC=0 -o src/f32-gemm/4x2-scalar.c
tools/xngen src/f32-gemm/scalar.c.in -D MR=4 -D NR=4 -D INC=0 -o src/f32-gemm/4x4-scalar.c

############################### AArch64 assembly ##############################
tools/xngen src/f32-gemm/1x12-aarch64-neonfma-cortex-a53.S.in -D INC=0 -o src/f32-gemm/1x12-aarch64-neonfma-cortex-a53.S
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-cortex-a57.S.in  -D INC=0 -o src/f32-gemm/1x8-aarch64-neonfma-cortex-a57.S
tools/xngen src/f32-gemm/1x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -o src/f32-gemm/1x8-aarch64-neonfma-cortex-a75.S
tools/xngen src/f32-gemm/4x12-aarch64-neonfma-cortex-a53.S.in -D INC=0 -o src/f32-gemm/4x12-aarch64-neonfma-cortex-a53.S
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-cortex-a57.S.in  -D INC=0 -o src/f32-gemm/4x8-aarch64-neonfma-cortex-a57.S
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -o src/f32-gemm/4x8-aarch64-neonfma-cortex-a75.S
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld128.S.in       -D INC=0 -o src/f32-gemm/4x8-aarch64-neonfma-ld128.S
tools/xngen src/f32-gemm/4x8-aarch64-neonfma-ld64.S.in        -D INC=0 -o src/f32-gemm/4x8-aarch64-neonfma-ld64.S
tools/xngen src/f32-gemm/5x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -o src/f32-gemm/5x8-aarch64-neonfma-cortex-a75.S
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a57.S.in  -D INC=0 -o src/f32-gemm/6x8-aarch64-neonfma-cortex-a57.S
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a73.S.in  -D INC=0 -o src/f32-gemm/6x8-aarch64-neonfma-cortex-a73.S
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -o src/f32-gemm/6x8-aarch64-neonfma-cortex-a75.S
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld64.S.in        -D INC=0 -o src/f32-gemm/6x8-aarch64-neonfma-ld64.S
tools/xngen src/f32-gemm/6x8-aarch64-neonfma-ld128.S.in       -D INC=0 -o src/f32-gemm/6x8-aarch64-neonfma-ld128.S

################################### ARM NEON ##################################
### LD64 micro-kernels
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=1 -D NR=8  -D FMA=1 -D INC=0 -o src/f32-gemm/1x8-neonfma-ld64.c
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=12 -D FMA=0 -D INC=0 -o src/f32-gemm/4x12-neon-ld64.c
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=12 -D FMA=1 -D INC=0 -o src/f32-gemm/4x12-neonfma-ld64.c
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=8  -D FMA=0 -D INC=0 -o src/f32-gemm/4x8-neon-ld64.c
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=4 -D NR=8  -D FMA=1 -D INC=0 -o src/f32-gemm/4x8-neonfma-ld64.c
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=5 -D NR=8  -D FMA=0 -D INC=0 -o src/f32-gemm/5x8-neon-ld64.c
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=5 -D NR=8  -D FMA=1 -D INC=0 -o src/f32-gemm/5x8-neonfma-ld64.c
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=6 -D NR=8  -D FMA=0 -D INC=0 -o src/f32-gemm/6x8-neon-ld64.c
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=6 -D NR=8  -D FMA=1 -D INC=0 -o src/f32-gemm/6x8-neonfma-ld64.c
### LD128 micro-kernels
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=0 -D INC=0 -o src/f32-gemm/4x8-neon-ld128.c
tools/xngen src/f32-gemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=1 -D INC=0 -o src/f32-gemm/4x8-neonfma-ld128.c
tools/xngen src/f32-gemm/neon-ld64.c.in      -D MR=1 -D NR=8  -D FMA=0 -D INC=0 -o src/f32-gemm/1x8-neon-ld64.c
### MRx2 micro-kernels
tools/xngen src/f32-gemm/MRx2-neon-ld64.c.in -D MR=4 -D NR=2  -D FMA=0 -D INC=0 -o src/f32-gemm/4x2-neon-ld64.c
tools/xngen src/f32-gemm/MRx2-neon-ld64.c.in -D MR=4 -D NR=2  -D FMA=1 -D INC=0 -o src/f32-gemm/4x2-neonfma-ld64.c

#################################### PSIMD ####################################
### LOAD1+BROADCAST micro-kernels
tools/xngen src/f32-gemm/psimd-loadsplat.c.in -D MR=1 -D NR=8 -D INC=0 -o src/f32-gemm/1x8-psimd-loadsplat.c
tools/xngen src/f32-gemm/psimd-loadsplat.c.in -D MR=4 -D NR=8 -D INC=0 -o src/f32-gemm/4x8-psimd-loadsplat.c
tools/xngen src/f32-gemm/psimd-loadsplat.c.in -D MR=6 -D NR=8 -D INC=0 -o src/f32-gemm/6x8-psimd-loadsplat.c
### LOAD4+DUPLICATE micro-kernels
tools/xngen src/f32-gemm/psimd-splat.c.in -D MR=1 -D NR=8 -D INC=0 -o src/f32-gemm/1x8-psimd-splat.c
tools/xngen src/f32-gemm/psimd-splat.c.in -D MR=4 -D NR=8 -D INC=0 -o src/f32-gemm/4x8-psimd-splat.c
tools/xngen src/f32-gemm/psimd-splat.c.in -D MR=6 -D NR=8 -D INC=0 -o src/f32-gemm/6x8-psimd-splat.c
### LOAD4+PERMUTE micro-kernels
tools/xngen src/f32-gemm/psimd-s4.c.in -D MR=1 -D NR=8 -D INC=0 -o src/f32-gemm/1x8s4-psimd.c
tools/xngen src/f32-gemm/psimd-s4.c.in -D MR=4 -D NR=8 -D INC=0 -o src/f32-gemm/4x8s4-psimd.c
tools/xngen src/f32-gemm/psimd-s4.c.in -D MR=6 -D NR=8 -D INC=0 -o src/f32-gemm/6x8s4-psimd.c

################################### x86 SSE ###################################
### LOAD1+BROADCAST micro-kernels
tools/xngen src/f32-gemm/sse-load1.c.in -D MR=1 -D NR=8 -D INC=0 -o src/f32-gemm/1x8-sse-load1.c
tools/xngen src/f32-gemm/sse-load1.c.in -D MR=4 -D NR=8 -D INC=0 -o src/f32-gemm/4x8-sse-load1.c
### LOAD4+DUPLICATE micro-kernels
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=1 -D NR=8 -D INC=0 -o src/f32-gemm/1x8-sse-dup.c
tools/xngen src/f32-gemm/sse-dup.c.in -D MR=4 -D NR=8 -D INC=0 -o src/f32-gemm/4x8-sse-dup.c
### LOAD4+PERMUTE micro-kernels
tools/xngen src/f32-gemm/sse-shuffle.c.in -D MR=1 -D NR=8 -D INC=0 -o src/f32-gemm/1x8s4-sse.c
tools/xngen src/f32-gemm/sse-shuffle.c.in -D MR=4 -D NR=8 -D INC=0 -o src/f32-gemm/4x8s4-sse.c


################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/f32-gemm.yaml --output test/f32-gemm.cc
