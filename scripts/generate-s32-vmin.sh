#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD VMIN #####################################
tools/xngen src/s32-vmin/s32-vmin.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vmin/gen/s32-vmin-scalar.c &
tools/xngen src/s32-vmin/s32-vmin.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s32-vmin/gen/s32-vmin-sse41.c &
tools/xngen src/s32-vmin/s32-vmin.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vmin/gen/s32-vmin-wasmsimd.c &
tools/xngen src/s32-vmin/s32-vmin.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vmin/gen/s32-vmin-neon.c &
tools/xngen src/s32-vmin/s32-vmin.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/s32-vmin/gen/s32-vmin-avx2.c &
tools/xngen src/s32-vmin/s32-vmin.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vmin/gen/s32-vmin-avx512f.c &

##################################### SIMD VMINC #####################################
tools/xngen src/s32-vmin/s32-vminc.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vmin/gen/s32-vminc-scalar.c &
tools/xngen src/s32-vmin/s32-vminc.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s32-vmin/gen/s32-vminc-sse41.c &
tools/xngen src/s32-vmin/s32-vminc.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vmin/gen/s32-vminc-wasmsimd.c &
tools/xngen src/s32-vmin/s32-vminc.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vmin/gen/s32-vminc-neon.c &
tools/xngen src/s32-vmin/s32-vminc.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/s32-vmin/gen/s32-vminc-avx2.c &
tools/xngen src/s32-vmin/s32-vminc.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vmin/gen/s32-vminc-avx512f.c &

wait
