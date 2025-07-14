#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ####################################
tools/xngen src/f32-vrsqrt/scalar-rsqrt.c.in -D BATCH_TILE=1 -o src/f32-vrsqrt/gen/f32-vrsqrt-scalar-rsqrt-u1.c &
tools/xngen src/f32-vrsqrt/scalar-rsqrt.c.in -D BATCH_TILE=2 -o src/f32-vrsqrt/gen/f32-vrsqrt-scalar-rsqrt-u2.c &
tools/xngen src/f32-vrsqrt/scalar-rsqrt.c.in -D BATCH_TILE=4 -o src/f32-vrsqrt/gen/f32-vrsqrt-scalar-rsqrt-u4.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vrsqrt/rvv.c.in -D LMUL=1 -o src/f32-vrsqrt/gen/f32-vrsqrt-rvv-rsqrt-u1v.c &
tools/xngen src/f32-vrsqrt/rvv.c.in -D LMUL=2 -o src/f32-vrsqrt/gen/f32-vrsqrt-rvv-rsqrt-u2v.c &
tools/xngen src/f32-vrsqrt/rvv.c.in -D LMUL=4 -o src/f32-vrsqrt/gen/f32-vrsqrt-rvv-rsqrt-u4v.c &

################################## SIMD rsqrt ##################################
tools/xngen src/f32-vrsqrt/simd-rsqrt.c.in -D ARCH=neon     -D BATCH_TILES=4,8,16   -o src/f32-vrsqrt/gen/f32-vrsqrt-neon-rsqrt.c &
tools/xngen src/f32-vrsqrt/simd-rsqrt.c.in -D ARCH=sse2     -D BATCH_TILES=4,8,16   -o src/f32-vrsqrt/gen/f32-vrsqrt-sse2-rsqrt.c &
tools/xngen src/f32-vrsqrt/simd-rsqrt.c.in -D ARCH=avx      -D BATCH_TILES=8,16,32  -o src/f32-vrsqrt/gen/f32-vrsqrt-avx-rsqrt.c &
tools/xngen src/f32-vrsqrt/simd-rsqrt.c.in -D ARCH=avx512f  -D BATCH_TILES=16,32,48 -o src/f32-vrsqrt/gen/f32-vrsqrt-avx512f-rsqrt.c &

################################ SIMD sqrt #####################################
tools/xngen src/f32-vrsqrt/simd-sqrt.c.in -D ARCH=scalar    -D BATCH_TILES=1,2,4    -o src/f32-vrsqrt/gen/f32-vrsqrt-scalar-sqrt.c &
tools/xngen src/f32-vrsqrt/simd-sqrt.c.in -D ARCH=sse2      -D BATCH_TILES=4,8,16   -o src/f32-vrsqrt/gen/f32-vrsqrt-sse2-sqrt.c &
tools/xngen src/f32-vrsqrt/simd-sqrt.c.in -D ARCH=avx       -D BATCH_TILES=8,16,32  -o src/f32-vrsqrt/gen/f32-vrsqrt-avx-sqrt.c &
tools/xngen src/f32-vrsqrt/simd-sqrt.c.in -D ARCH=avx512f   -D BATCH_TILES=16,32,48 -o src/f32-vrsqrt/gen/f32-vrsqrt-avx512f-sqrt.c &

wait
