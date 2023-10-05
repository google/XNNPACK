#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-ppmm/scalar.c.in -D MR=4 -D NR=4 -o src/f32-ppmm/gen/f32-ppmm-4x4-minmax-scalar.c &
tools/xngen src/f32-ppmm/scalar.c.in -D MR=2 -D NR=4 -o src/f32-ppmm/gen/f32-ppmm-2x4-minmax-scalar.c &
tools/xngen src/f32-ppmm/scalar.c.in -D MR=4 -D NR=2 -o src/f32-ppmm/gen/f32-ppmm-4x2-minmax-scalar.c &
tools/xngen src/f32-ppmm/scalar.c.in -D MR=3 -D NR=3 -o src/f32-ppmm/gen/f32-ppmm-3x3-minmax-scalar.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-ppmm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D MINMAX=MINMAX  -o src/f32-ppmm/gen/f32-ppmm-4x8-minmax-wasmsimd-arm-splat.c &
tools/xngen src/f32-ppmm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D MINMAX=PMINMAX -o src/f32-ppmm/gen/f32-ppmm-4x8-minmax-wasmsimd-x86-splat.c &

################################### ARM NEON ##################################
tools/xngen src/f32-ppmm/neon.c.in -D MR=4 -D NR=8  -D FMA=0 -D PREFETCH=0 -o src/f32-ppmm/gen/f32-ppmm-4x8-minmax-neon.c &
tools/xngen src/f32-ppmm/neon.c.in -D MR=4 -D NR=8  -D FMA=1 -D PREFETCH=0 -o src/f32-ppmm/gen/f32-ppmm-4x8-minmax-aarch64-neonfma.c &
tools/xngen src/f32-ppmm/neon.c.in -D MR=8 -D NR=8  -D FMA=0 -D PREFETCH=0 -o src/f32-ppmm/gen/f32-ppmm-8x8-minmax-neon.c &
tools/xngen src/f32-ppmm/neon.c.in -D MR=8 -D NR=8  -D FMA=1 -D PREFETCH=0 -o src/f32-ppmm/gen/f32-ppmm-8x8-minmax-aarch64-neonfma.c &
tools/xngen src/f32-ppmm/neon.c.in -D MR=4 -D NR=8  -D FMA=0 -D PREFETCH=1 -o src/f32-ppmm/gen/f32-ppmm-4x8-minmax-neon-prfm.c &
tools/xngen src/f32-ppmm/neon.c.in -D MR=4 -D NR=8  -D FMA=1 -D PREFETCH=1 -o src/f32-ppmm/gen/f32-ppmm-4x8-minmax-aarch64-neonfma-prfm.c &
tools/xngen src/f32-ppmm/neon.c.in -D MR=8 -D NR=8  -D FMA=0 -D PREFETCH=1 -o src/f32-ppmm/gen/f32-ppmm-8x8-minmax-neon-prfm.c &
tools/xngen src/f32-ppmm/neon.c.in -D MR=8 -D NR=8  -D FMA=1 -D PREFETCH=1 -o src/f32-ppmm/gen/f32-ppmm-8x8-minmax-aarch64-neonfma-prfm.c &
tools/xngen src/f32-ppmm/neon.c.in -D MR=4 -D NR=16 -D FMA=0 -D PREFETCH=0 -o src/f32-ppmm/gen/f32-ppmm-4x16-minmax-neon.c &
tools/xngen src/f32-ppmm/neon.c.in -D MR=4 -D NR=16 -D FMA=1 -D PREFETCH=0 -o src/f32-ppmm/gen/f32-ppmm-4x16-minmax-aarch64-neonfma.c &
tools/xngen src/f32-ppmm/neon.c.in -D MR=4 -D NR=16 -D FMA=0 -D PREFETCH=1 -o src/f32-ppmm/gen/f32-ppmm-4x16-minmax-neon-prfm.c &
tools/xngen src/f32-ppmm/neon.c.in -D MR=4 -D NR=16 -D FMA=1 -D PREFETCH=1 -o src/f32-ppmm/gen/f32-ppmm-4x16-minmax-aarch64-neonfma-prfm.c &

############################### AArch64 assembly ##############################
### LD128 micro-kernels
tools/xngen src/f32-ppmm/4x8-aarch64-neonfma-ld128.S.in -D PREFETCH=0 -o src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-ld128.S &
tools/xngen src/f32-ppmm/4x8-aarch64-neonfma-ld128.S.in -D PREFETCH=1 -o src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-ld128-prfm.S &
tools/xngen src/f32-ppmm/8x8-aarch64-neonfma-ld128.S.in -D PREFETCH=0 -o src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-ld128.S &
tools/xngen src/f32-ppmm/8x8-aarch64-neonfma-ld128.S.in -D PREFETCH=1 -o src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-ld128-prfm.S &
### Cortex A75 micro-kernels
tools/xngen src/f32-ppmm/4x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=0 -o src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-ppmm/4x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=1 -o src/f32-ppmm/gen/f32-ppmm-4x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S &
tools/xngen src/f32-ppmm/8x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=0 -o src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-ppmm/8x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=1 -o src/f32-ppmm/gen/f32-ppmm-8x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S &

################################### x86 SSE ###################################
tools/xngen src/f32-ppmm/sse.c.in -D MR=4 -D NR=8 -o src/f32-ppmm/gen/f32-ppmm-4x8-minmax-sse.c &

wait
