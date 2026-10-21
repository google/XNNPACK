#!/bin/sh
# Copyright 2023-2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################# SIMD Wrappers ################################
tools/xngen src/f32-rsum2/simd.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16 -o src/f32-rsum2/gen/f32-rsum2-neon.c &
tools/xngen src/f32-rsum2/simd.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16 -o src/f32-rsum2/gen/f32-rsum2-sse2-u4.c &
tools/xngen src/f32-rsum2/simd.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32 -o src/f32-rsum2/gen/f32-rsum2-avx-u8.c &
tools/xngen src/f32-rsum2/simd.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -o src/f32-rsum2/gen/f32-rsum2-avx512f-u16.c &
tools/xngen src/f32-rsum2/simd.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16 -o src/f32-rsum2/gen/f32-rsum2-wasmsimd-u4.c &
tools/xngen src/f32-rsum2/simd.c.in -D ARCH=scalar -D BATCH_TILES=1,2,3,4 -o src/f32-rsum2/gen/f32-rsum2-scalar-u1.c &

################################## RISC-V Vector ##############################
tools/xngen src/f32-rsum2/rvv.c.in -D LMUL=4 -o src/f32-rsum2/gen/f32-rsum2-rvv-u4v.c &
tools/xngen src/f32-rsum2/rvv.c.in -D LMUL=8 -o src/f32-rsum2/gen/f32-rsum2-rvv-u8v.c &

################################## Hexagon HVX ################################
tools/xngen src/f32-rsum2/hvx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=1 -o src/f32-rsum2/gen/f32-rsum2-hvx-u32.c &
tools/xngen src/f32-rsum2/hvx.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -o src/f32-rsum2/gen/f32-rsum2-hvx-u64-acc2.c &
tools/xngen src/f32-rsum2/hvx.c.in -D BATCH_TILE=96 -D ACCUMULATORS=3 -o src/f32-rsum2/gen/f32-rsum2-hvx-u96-acc3.c &
tools/xngen src/f32-rsum2/hvx.c.in -D BATCH_TILE=128 -D ACCUMULATORS=2 -o src/f32-rsum2/gen/f32-rsum2-hvx-u128-acc2.c &
tools/xngen src/f32-rsum2/hvx.c.in -D BATCH_TILE=128 -D ACCUMULATORS=4 -o src/f32-rsum2/gen/f32-rsum2-hvx-u128-acc4.c &

wait
