#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD VMUL #####################################
# tools/xngen src/s16-vmul/s16-vmul.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s16-vmul/gen/s16-vmul-scalar.c &
# tools/xngen src/s16-vmul/s16-vmul.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s16-vmul/gen/s16-vmul-sse41.c &
# tools/xngen src/s16-vmul/s16-vmul.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s16-vmul/gen/s16-vmul-wasmsimd.c &
# tools/xngen src/s16-vmul/s16-vmul.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s16-vmul/gen/s16-vmul-neon.c &
tools/xngen src/s16-vmul/s16-vmul.c.in -D ARCH=avx2 -D BATCH_TILES=16,32  -o src/s16-vmul/gen/s16-vmul-avx2.c &
# tools/xngen src/s16-vmul/s16-vmul.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s16-vmul/gen/s16-vmul-avx512f.c &

##################################### SIMD VMULC #####################################
# tools/xngen src/s16-vmul/s16-vmulc.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s16-vmul/gen/s16-vmulc-scalar.c &
# tools/xngen src/s16-vmul/s16-vmulc.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s16-vmul/gen/s16-vmulc-sse41.c &
# tools/xngen src/s16-vmul/s16-vmulc.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s16-vmul/gen/s16-vmulc-wasmsimd.c &
# tools/xngen src/s16-vmul/s16-vmulc.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s16-vmul/gen/s16-vmulc-neon.c &
tools/xngen src/s16-vmul/s16-vmulc.c.in -D ARCH=avx2 -D BATCH_TILES=16,32  -o src/s16-vmul/gen/s16-vmulc-avx2.c &
# tools/xngen src/s16-vmul/s16-vmulc.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s16-vmul/gen/s16-vmulc-avx512f.c &

wait