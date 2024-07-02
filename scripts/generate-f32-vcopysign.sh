#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f32-vcopysign/copysign.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/f32-vcopysign/gen/f32-vcopysign-scalar.c &
tools/xngen src/f32-vcopysign/copysign.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16  -o src/f32-vcopysign/gen/f32-vcopysign-sse2.c &
tools/xngen src/f32-vcopysign/copysign.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/f32-vcopysign/gen/f32-vcopysign-wasmsimd.c &
tools/xngen src/f32-vcopysign/copysign.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/f32-vcopysign/gen/f32-vcopysign-neon.c &
tools/xngen src/f32-vcopysign/copysign.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32  -o src/f32-vcopysign/gen/f32-vcopysign-avx.c &
tools/xngen src/f32-vcopysign/copysign.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/f32-vcopysign/gen/f32-vcopysign-avx512f.c &

# Scalar sign
tools/xngen src/f32-vcopysign/copysignc.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/f32-vcopysign/gen/f32-vcopysignc-scalar.c &
tools/xngen src/f32-vcopysign/copysignc.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16  -o src/f32-vcopysign/gen/f32-vcopysignc-sse2.c &
tools/xngen src/f32-vcopysign/copysignc.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/f32-vcopysign/gen/f32-vcopysignc-wasmsimd.c &
tools/xngen src/f32-vcopysign/copysignc.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/f32-vcopysign/gen/f32-vcopysignc-neon.c &
tools/xngen src/f32-vcopysign/copysignc.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32  -o src/f32-vcopysign/gen/f32-vcopysignc-avx.c &
tools/xngen src/f32-vcopysign/copysignc.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/f32-vcopysign/gen/f32-vcopysignc-avx512f.c &

# Scalar mag
tools/xngen src/f32-vcopysign/rcopysignc.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/f32-vcopysign/gen/f32-vrcopysignc-scalar.c &
tools/xngen src/f32-vcopysign/rcopysignc.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16  -o src/f32-vcopysign/gen/f32-vrcopysignc-sse2.c &
tools/xngen src/f32-vcopysign/rcopysignc.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/f32-vcopysign/gen/f32-vrcopysignc-wasmsimd.c &
tools/xngen src/f32-vcopysign/rcopysignc.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/f32-vcopysign/gen/f32-vrcopysignc-neon.c &
tools/xngen src/f32-vcopysign/rcopysignc.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32  -o src/f32-vcopysign/gen/f32-vrcopysignc-avx.c &
tools/xngen src/f32-vcopysign/rcopysignc.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/f32-vcopysign/gen/f32-vrcopysignc-avx512f.c &

wait
