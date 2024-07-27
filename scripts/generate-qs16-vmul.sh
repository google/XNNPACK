#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD VMUL #####################################
python3 tools/xngen src/qs16-vmul/qs16-vmul.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/qs16-vmul/gen/qs16-vmul-minmax-scalar.c &
python3 tools/xngen src/qs16-vmul/qs16-vmul.c.in -D ARCH=sse41 -D BATCH_TILES=8,16  -o src/qs16-vmul/gen/qs16-vmul-minmax-sse41.c &
python3 tools/xngen src/qs16-vmul/qs16-vmul.c.in -D ARCH=wasmsimd -D BATCH_TILES=8,16  -o src/qs16-vmul/gen/qs16-vmul-minmax-wasmsimd.c &
python3 tools/xngen src/qs16-vmul/qs16-vmul.c.in -D ARCH=neon -D BATCH_TILES=8,16  -o src/qs16-vmul/gen/qs16-vmul-minmax-neon.c &
python3 tools/xngen src/qs16-vmul/qs16-vmul.c.in -D ARCH=avx2 -D BATCH_TILES=16,32  -o src/qs16-vmul/gen/qs16-vmul-minmax-avx2.c &
python3 tools/xngen src/qs16-vmul/qs16-vmul.c.in -D ARCH=avx512bw -D BATCH_TILES=32,64  -o src/qs16-vmul/gen/qs16-vmul-minmax-avx512skx.c &

##################################### SIMD VMULC #####################################
python3 tools/xngen src/qs16-vmul/qs16-vmulc.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/qs16-vmul/gen/qs16-vmulc-minmax-scalar.c &
python3 tools/xngen src/qs16-vmul/qs16-vmulc.c.in -D ARCH=sse41 -D BATCH_TILES=8,16  -o src/qs16-vmul/gen/qs16-vmulc-minmax-sse41.c &
python3 tools/xngen src/qs16-vmul/qs16-vmulc.c.in -D ARCH=wasmsimd -D BATCH_TILES=8,16  -o src/qs16-vmul/gen/qs16-vmulc-minmax-wasmsimd.c &
python3 tools/xngen src/qs16-vmul/qs16-vmulc.c.in -D ARCH=neon -D BATCH_TILES=8,16  -o src/qs16-vmul/gen/qs16-vmulc-minmax-neon.c &
python3 tools/xngen src/qs16-vmul/qs16-vmulc.c.in -D ARCH=avx2 -D BATCH_TILES=16,32  -o src/qs16-vmul/gen/qs16-vmulc-minmax-avx2.c &
python3 tools/xngen src/qs16-vmul/qs16-vmulc.c.in -D ARCH=avx512bw -D BATCH_TILES=32,64  -o src/qs16-vmul/gen/qs16-vmulc-minmax-avx512skx.c &

wait