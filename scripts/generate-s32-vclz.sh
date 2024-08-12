#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD VCLZ #####################################
tools/xngen src/s32-vclz/s32-vclz.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vclz/gen/s32-vclz-scalar.c &
tools/xngen src/s32-vclz/s32-vclz.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s32-vclz/gen/s32-vclz-sse41.c &
#tools/xngen src/s32-vclz/s32-vclz.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vclz/gen/s32-vclz-wasmsimd.c &
tools/xngen src/s32-vclz/s32-vclz.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vclz/gen/s32-vclz-neon.c &
tools/xngen src/s32-vclz/s32-vclz.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/s32-vclz/gen/s32-vclz-avx2.c &
tools/xngen src/s32-vclz/s32-vclz.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vclz/gen/s32-vclz-avx512f.c &

wait
