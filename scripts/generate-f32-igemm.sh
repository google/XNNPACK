#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-igemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -o src/f32-igemm/gen/1x4-scalar.c
tools/xngen src/f32-igemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -o src/f32-igemm/gen/2x4-scalar.c
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -o src/f32-igemm/gen/4x2-scalar.c
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -o src/f32-igemm/gen/4x4-scalar.c

### WAsm-specific micro-kernels
tools/xngen src/f32-igemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=1 -o src/f32-igemm/gen/1x4-wasm.c
tools/xngen src/f32-igemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=1 -o src/f32-igemm/gen/2x4-wasm.c
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=1 -o src/f32-igemm/gen/4x2-wasm.c
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=1 -o src/f32-igemm/gen/4x4-wasm.c

############################### AArch64 assembly ##############################
# Cortex A75 / A57 micro-kernels
tools/xngen src/f32-igemm/1x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=0 -o src/f32-igemm/gen/1x8-aarch64-neonfma-cortex-a57.S
tools/xngen src/f32-igemm/1x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=1 -o src/f32-igemm/gen/1x8-aarch64-neonfma-cortex-a75.S
tools/xngen src/f32-igemm/4x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=0 -o src/f32-igemm/gen/4x8-aarch64-neonfma-cortex-a57.S
tools/xngen src/f32-igemm/4x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=1 -o src/f32-igemm/gen/4x8-aarch64-neonfma-cortex-a75.S
tools/xngen src/f32-igemm/5x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=0 -o src/f32-igemm/gen/5x8-aarch64-neonfma-cortex-a57.S
tools/xngen src/f32-igemm/5x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=1 -o src/f32-igemm/gen/5x8-aarch64-neonfma-cortex-a75.S
tools/xngen src/f32-igemm/6x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=0 -o src/f32-igemm/gen/6x8-aarch64-neonfma-cortex-a57.S
tools/xngen src/f32-igemm/6x8-aarch64-neonfma-cortex-a75.S.in  -D INC=0 -D PREFETCH=1 -o src/f32-igemm/gen/6x8-aarch64-neonfma-cortex-a75.S

############################### AArch32 assembly ##############################
tools/xngen src/f32-igemm/4x8-aarch32-neon-cortex-a75.S.in  -D INC=0 -D PREFETCH=0 -o src/f32-igemm/gen/4x8-aarch32-neon-cortex-a75.S
tools/xngen src/f32-igemm/4x8-aarch32-neon-cortex-a75.S.in  -D INC=0 -D PREFETCH=1 -o src/f32-igemm/gen/4x8-aarch32-neon-pld-cortex-a75.S

################################### ARM NEON ##################################
### LD64 micro-kernels
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=1 -D NR=8 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/1x8-neon-lane-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=1 -D NR=8 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/1x8-neonfma-lane-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=4 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/4x4-neon-lane-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=4 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/4x4-neonfma-lane-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=8 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/4x8-neon-lane-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=8 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/4x8-neonfma-lane-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=6 -D NR=8 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/6x8-neon-lane-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=6 -D NR=8 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/6x8-neonfma-lane-ld64.c
### LD128 micro-kernels
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=4 -D NR=8 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/4x8-neon-lane-ld128.c
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=4 -D NR=8 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/4x8-neonfma-lane-ld128.c
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=6 -D NR=8 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/6x8-neon-lane-ld128.c
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=6 -D NR=8 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/6x8-neonfma-lane-ld128.c
### MRx2 micro-kernels-
tools/xngen src/f32-igemm/MRx2-neon-ld64.c.in -D MR=4 -D NR=2 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/4x2-neon-lane-ld64.c
tools/xngen src/f32-igemm/MRx2-neon-ld64.c.in -D MR=4 -D NR=2 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/4x2-neonfma-lane-ld64.c
### DUP LD64 micro-kernels
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=1 -D NR=8 -D FMA=0 -D DUP=1 -o src/f32-igemm/gen/1x8-neon-dup-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=1 -D NR=8 -D FMA=1 -D DUP=1 -o src/f32-igemm/gen/1x8-neonfma-dup-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=8 -D FMA=0 -D DUP=1 -o src/f32-igemm/gen/4x8-neon-dup-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=8 -D FMA=1 -D DUP=1 -o src/f32-igemm/gen/4x8-neonfma-dup-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=6 -D NR=8 -D FMA=0 -D DUP=1 -o src/f32-igemm/gen/6x8-neon-dup-ld64.c
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=6 -D NR=8 -D FMA=1 -D DUP=1 -o src/f32-igemm/gen/6x8-neonfma-dup-ld64.c
### DUP LD128 micro-kernels
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=4 -D NR=8 -D FMA=0 -D DUP=1 -o src/f32-igemm/gen/4x8-neon-dup-ld128.c
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=4 -D NR=8 -D FMA=1 -D DUP=1 -o src/f32-igemm/gen/4x8-neonfma-dup-ld128.c
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=6 -D NR=8 -D FMA=0 -D DUP=1 -o src/f32-igemm/gen/6x8-neon-dup-ld128.c
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=6 -D NR=8 -D FMA=1 -D DUP=1 -o src/f32-igemm/gen/6x8-neonfma-dup-ld128.c
### LOAD4+PERMUTE micro-kernels
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=1 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/1x8s4-neon.c
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=1 -D NR=8 -D FMA=1 -o src/f32-igemm/gen/1x8s4-neonfma.c
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=4 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/4x8s4-neon.c
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=4 -D NR=8 -D FMA=1 -o src/f32-igemm/gen/4x8s4-neonfma.c
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=6 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/6x8s4-neon.c
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=6 -D NR=8 -D FMA=1 -o src/f32-igemm/gen/6x8s4-neonfma.c
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=8 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/8x8s4-neon.c
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=8 -D NR=8 -D FMA=1 -o src/f32-igemm/gen/8x8s4-neonfma.c

#################################### PSIMD ####################################
### LOAD1+BROADCAST micro-kernels
tools/xngen src/f32-igemm/psimd-loadsplat.c.in -D MR=1 -D NR=8 -o src/f32-igemm/gen/1x8-psimd-loadsplat.c
tools/xngen src/f32-igemm/psimd-loadsplat.c.in -D MR=4 -D NR=8 -o src/f32-igemm/gen/4x8-psimd-loadsplat.c
tools/xngen src/f32-igemm/psimd-loadsplat.c.in -D MR=6 -D NR=8 -o src/f32-igemm/gen/6x8-psimd-loadsplat.c
### LOAD4+DUPLICATE micro-kernels
tools/xngen src/f32-igemm/psimd-splat.c.in -D MR=1 -D NR=8 -o src/f32-igemm/gen/1x8-psimd-splat.c
tools/xngen src/f32-igemm/psimd-splat.c.in -D MR=4 -D NR=8 -o src/f32-igemm/gen/4x8-psimd-splat.c
tools/xngen src/f32-igemm/psimd-splat.c.in -D MR=6 -D NR=8 -o src/f32-igemm/gen/6x8-psimd-splat.c
### LOAD4+PERMUTE micro-kernels
tools/xngen src/f32-igemm/psimd-s4.c.in -D MR=1 -D NR=8 -o src/f32-igemm/gen/1x8s4-psimd.c
tools/xngen src/f32-igemm/psimd-s4.c.in -D MR=4 -D NR=8 -o src/f32-igemm/gen/4x8s4-psimd.c
tools/xngen src/f32-igemm/psimd-s4.c.in -D MR=6 -D NR=8 -o src/f32-igemm/gen/6x8s4-psimd.c
### MRx2 micro-kernels
tools/xngen src/f32-igemm/MRx2c4-psimd.c.in -D MR=4 -D NR=2 -o src/f32-igemm/gen/4x2c4-psimd.c

################################### x86 SSE ###################################
### LOAD1+BROADCAST micro-kernels
tools/xngen src/f32-igemm/sse-load1.c.in -D MR=1 -D NR=8 -o src/f32-igemm/gen/1x8-sse-load1.c
tools/xngen src/f32-igemm/sse-load1.c.in -D MR=4 -D NR=8 -o src/f32-igemm/gen/4x8-sse-load1.c
### LOAD4+DUPLICATE micro-kernels
tools/xngen src/f32-igemm/sse-dup.c.in -D MR=1 -D NR=8 -o src/f32-igemm/gen/1x8-sse-dup.c
tools/xngen src/f32-igemm/sse-dup.c.in -D MR=4 -D NR=8 -o src/f32-igemm/gen/4x8-sse-dup.c
### LOAD4+PERMUTE micro-kernels
tools/xngen src/f32-igemm/sse-shuffle.c.in -D MR=1 -D NR=8 -o src/f32-igemm/gen/1x8s4-sse.c
tools/xngen src/f32-igemm/sse-shuffle.c.in -D MR=4 -D NR=8 -o src/f32-igemm/gen/4x8s4-sse.c
### MRx2 micro-kernels
tools/xngen src/f32-igemm/MRx2c4-sse.c.in -D MR=4 -D NR=2 -o src/f32-igemm/gen/4x2c4-sse.c

################################### x86 AVX ###################################
### AVX+BROADCAST micro-kernels
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=1 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/1x8-avx-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=4 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/4x8-avx-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=5 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/5x8-avx-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=6 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/6x8-avx-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=7 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/7x8-avx-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=1 -D NR=16 -D FMA=0 -o src/f32-igemm/gen/1x16-avx-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=3 -D NR=16 -D FMA=0 -o src/f32-igemm/gen/3x16-avx-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=4 -D NR=16 -D FMA=0 -o src/f32-igemm/gen/4x16-avx-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D FMA=0 -o src/f32-igemm/gen/5x16-avx-broadcast.c
### FMA3+BROADCAST micro-kernels
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=1 -D NR=8 -D FMA=3 -o src/f32-igemm/gen/1x8-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=4 -D NR=8 -D FMA=3 -o src/f32-igemm/gen/4x8-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=5 -D NR=8 -D FMA=3 -o src/f32-igemm/gen/5x8-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=6 -D NR=8 -D FMA=3 -o src/f32-igemm/gen/6x8-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=7 -D NR=8 -D FMA=3 -o src/f32-igemm/gen/7x8-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=8 -D NR=8 -D FMA=3 -o src/f32-igemm/gen/8x8-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=1 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/1x16-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=3 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/3x16-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=4 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/4x16-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/5x16-fma3-broadcast.c

tools/xngen src/f32-igemm/avx-shuffle4.c.in -D MR=1 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/1x16s4-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-shuffle4.c.in -D MR=3 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/3x16s4-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-shuffle4.c.in -D MR=4 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/4x16s4-fma3-broadcast.c
tools/xngen src/f32-igemm/avx-shuffle4.c.in -D MR=5 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/5x16s4-fma3-broadcast.c

################################# x86 AVX-512 #################################
### AVX512F+BROADCAST micro-kernels
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=1 -D NR=16 -o src/f32-igemm/gen/1x16-avx512f-broadcast.c
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=4 -D NR=16 -o src/f32-igemm/gen/4x16-avx512f-broadcast.c
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=5 -D NR=16 -o src/f32-igemm/gen/5x16-avx512f-broadcast.c
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=6 -D NR=16 -o src/f32-igemm/gen/6x16-avx512f-broadcast.c
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=7 -D NR=16 -o src/f32-igemm/gen/7x16-avx512f-broadcast.c
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=8 -D NR=16 -o src/f32-igemm/gen/8x16-avx512f-broadcast.c

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/f32-igemm.yaml --output test/f32-igemm.cc
