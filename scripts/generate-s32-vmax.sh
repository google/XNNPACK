#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD VMAX #####################################
tools/xngen src/s32-vmax/s32-vmax.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vmax/gen/s32-vmax-scalar.c &
tools/xngen src/s32-vmax/s32-vmax.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s32-vmax/gen/s32-vmax-sse41.c &
tools/xngen src/s32-vmax/s32-vmax.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vmax/gen/s32-vmax-wasmsimd.c &
tools/xngen src/s32-vmax/s32-vmax.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vmax/gen/s32-vmax-neon.c &
tools/xngen src/s32-vmax/s32-vmax.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/s32-vmax/gen/s32-vmax-avx2.c &
tools/xngen src/s32-vmax/s32-vmax.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vmax/gen/s32-vmax-avx512f.c &

##################################### SIMD VMAXC #####################################
tools/xngen src/s32-vmax/s32-vmaxc.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/s32-vmax/gen/s32-vmaxc-scalar.c &
tools/xngen src/s32-vmax/s32-vmaxc.c.in -D ARCH=sse41 -D BATCH_TILES=4,8,12,16  -o src/s32-vmax/gen/s32-vmaxc-sse41.c &
tools/xngen src/s32-vmax/s32-vmaxc.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/s32-vmax/gen/s32-vmaxc-wasmsimd.c &
tools/xngen src/s32-vmax/s32-vmaxc.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/s32-vmax/gen/s32-vmaxc-neon.c &
tools/xngen src/s32-vmax/s32-vmaxc.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/s32-vmax/gen/s32-vmaxc-avx2.c &
tools/xngen src/s32-vmax/s32-vmaxc.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/s32-vmax/gen/s32-vmaxc-avx512f.c &

wait
