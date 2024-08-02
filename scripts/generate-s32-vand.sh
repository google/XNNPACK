#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD VAND #####################################
tools/xngen src/s32-vand/s32-vand.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vand/gen/s32-vand-scalar.c &
tools/xngen src/s32-vand/s32-vand.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s32-vand/gen/s32-vand-sse41.c &
tools/xngen src/s32-vand/s32-vand.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vand/gen/s32-vand-wasmsimd.c &
tools/xngen src/s32-vand/s32-vand.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vand/gen/s32-vand-neon.c &
tools/xngen src/s32-vand/s32-vand.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/s32-vand/gen/s32-vand-avx2.c &
tools/xngen src/s32-vand/s32-vand.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vand/gen/s32-vand-avx512f.c &

##################################### SIMD VANDC #####################################
tools/xngen src/s32-vand/s32-vandc.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vand/gen/s32-vandc-scalar.c &
tools/xngen src/s32-vand/s32-vandc.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s32-vand/gen/s32-vandc-sse41.c &
tools/xngen src/s32-vand/s32-vandc.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vand/gen/s32-vandc-wasmsimd.c &
tools/xngen src/s32-vand/s32-vandc.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vand/gen/s32-vandc-neon.c &
tools/xngen src/s32-vand/s32-vandc.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/s32-vand/gen/s32-vandc-avx2.c &
tools/xngen src/s32-vand/s32-vandc.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vand/gen/s32-vandc-avx512f.c &

wait
