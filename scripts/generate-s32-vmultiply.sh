#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
python3 tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vmultiply/gen/s32-vmultiply-scalar.c &
python3 tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16  -o src/s32-vmultiply/gen/s32-vmultiply-sse2.c &
python3 tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vmultiply/gen/s32-vmultiply-wasmsimd.c &
python3 tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vmultiply/gen/s32-vmultiply-neon.c &
python3 tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32  -o src/s32-vmultiply/gen/s32-vmultiply-avx.c &
python3 tools/xngen src/s32-vmultiply/s32-vmultiply.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vmultiply/gen/s32-vmultiply-avx512f.c &

wait
