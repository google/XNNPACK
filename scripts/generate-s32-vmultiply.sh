#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vmultiply/gen/s32-vmultiply-scalar.c &
tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s32-vmultiply/gen/s32-vmultiply-sse41.c &
tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vmultiply/gen/s32-vmultiply-wasmsimd.c &
tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vmultiply/gen/s32-vmultiply-neon.c &
tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/s32-vmultiply/gen/s32-vmultiply-avx2.c &
tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vmultiply/gen/s32-vmultiply-avx512f.c &


tools/xngen src/s32-vmultiply/s32-vmultiplyc.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vmultiply/gen/s32-vmultiplyc-scalar.c &
tools/xngen src/s32-vmultiply/s32-vmultiplyc.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s32-vmultiply/gen/s32-vmultiplyc-sse41.c &
tools/xngen src/s32-vmultiply/s32-vmultiplyc.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vmultiply/gen/s32-vmultiplyc-wasmsimd.c &
tools/xngen src/s32-vmultiply/s32-vmultiplyc.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vmultiply/gen/s32-vmultiplyc-neon.c &
tools/xngen src/s32-vmultiply/s32-vmultiplyc.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/s32-vmultiply/gen/s32-vmultiplyc-avx2.c &
tools/xngen src/s32-vmultiply/s32-vmultiplyc.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vmultiply/gen/s32-vmultiplyc-avx512f.c &

wait
