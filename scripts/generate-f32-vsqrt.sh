#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################ RISC-V Vector ################################
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=1 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u1v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=2 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u2v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=4 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u4v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=8 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u8v.c &

##################################### SIMD sqrt ################################
tools/xngen src/f32-vsqrt/simd-sqrt.c.in -D ARCH=scalar   -D BATCH_TILES=1,2,4,8   -o src/f32-vsqrt/gen/f32-vsqrt-scalar-sqrt.c &
tools/xngen src/f32-vsqrt/simd-sqrt.c.in -D ARCH=sse2     -D BATCH_TILES=4,8,16    -o src/f32-vsqrt/gen/f32-vsqrt-sse2-sqrt.c &
tools/xngen src/f32-vsqrt/simd-sqrt.c.in -D ARCH=avx      -D BATCH_TILES=8,16,32   -o src/f32-vsqrt/gen/f32-vsqrt-avx-sqrt.c &
tools/xngen src/f32-vsqrt/simd-sqrt.c.in -D ARCH=avx512f  -D BATCH_TILES=16,32,48 -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-sqrt.c &
tools/xngen src/f32-vsqrt/simd-sqrt.c.in -D ARCH=neon     -D BATCH_TILES=4,8,16  -o src/f32-vsqrt/gen/f32-vsqrt-aarch64-neon-sqrt.c &
tools/xngen src/f32-vsqrt/simd-sqrt.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,16  -o src/f32-vsqrt/gen/f32-vsqrt-wasmsimd-sqrt.c &

################################## SIMD rsqrt ##################################
tools/xngen src/f32-vsqrt/simd-rsqrt.c.in -D ARCH=neon    -D BATCH_TILES=4,8,16   -o src/f32-vsqrt/gen/f32-vsqrt-neon-rsqrt.c &
tools/xngen src/f32-vsqrt/simd-rsqrt.c.in -D ARCH=sse2    -D BATCH_TILES=4,8,16   -o src/f32-vsqrt/gen/f32-vsqrt-sse2-rsqrt.c &
tools/xngen src/f32-vsqrt/simd-rsqrt.c.in -D ARCH=avx     -D BATCH_TILES=8,16,32  -o src/f32-vsqrt/gen/f32-vsqrt-avx-rsqrt.c &
tools/xngen src/f32-vsqrt/simd-rsqrt.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48 -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-rsqrt.c &

wait
