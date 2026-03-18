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

wait
