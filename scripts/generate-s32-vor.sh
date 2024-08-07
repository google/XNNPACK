#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD VAND #####################################
tools/xngen src/s32-vor/s32-vor.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vor/gen/s32-vor-scalar.c &
tools/xngen src/s32-vor/s32-vor.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s32-vor/gen/s32-vor-sse41.c &
tools/xngen src/s32-vor/s32-vor.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vor/gen/s32-vor-wasmsimd.c &
tools/xngen src/s32-vor/s32-vor.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vor/gen/s32-vor-neon.c &
tools/xngen src/s32-vor/s32-vor.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/s32-vor/gen/s32-vor-avx2.c &
tools/xngen src/s32-vor/s32-vor.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vor/gen/s32-vor-avx512f.c &

##################################### SIMD VANDC #####################################
tools/xngen src/s32-vor/s32-vorc.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vor/gen/s32-vorc-scalar.c &
tools/xngen src/s32-vor/s32-vorc.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s32-vor/gen/s32-vorc-sse41.c &
tools/xngen src/s32-vor/s32-vorc.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vor/gen/s32-vorc-wasmsimd.c &
tools/xngen src/s32-vor/s32-vorc.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vor/gen/s32-vorc-neon.c &
tools/xngen src/s32-vor/s32-vorc.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/s32-vor/gen/s32-vorc-avx2.c &
tools/xngen src/s32-vor/s32-vorc.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vor/gen/s32-vorc-avx512f.c &

wait
