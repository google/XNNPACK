#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f32-vsin/rational-5-4.c.in -D FUN=COS -D ARCH=scalar   -D BATCH_TILES=1,2,4,8      -D DIV=DIV -o src/f32-vcos/gen/f32-vcos-scalar-rational-5-4-div.c &
tools/xngen src/f32-vsin/rational-5-4.c.in -D FUN=COS -D ARCH=sse2     -D BATCH_TILES=4,8,12,16    -D DIV=DIV -o src/f32-vcos/gen/f32-vcos-sse2-rational-5-4-div.c &
tools/xngen src/f32-vsin/rational-5-4.c.in -D FUN=COS -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16    -D DIV=DIV -o src/f32-vcos/gen/f32-vcos-wasmsimd-rational-5-4-div.c &
tools/xngen src/f32-vsin/rational-5-4.c.in -D FUN=COS -D ARCH=neon     -D BATCH_TILES=4,8,12,16    -D DIV=DIV -o src/f32-vcos/gen/f32-vcos-neon-rational-5-4-div.c &
tools/xngen src/f32-vsin/rational-5-4.c.in -D FUN=COS -D ARCH=avx      -D BATCH_TILES=8,16,24,32   -D DIV=DIV -o src/f32-vcos/gen/f32-vcos-avx-rational-5-4-div.c &
tools/xngen src/f32-vsin/rational-5-4.c.in -D FUN=COS -D ARCH=fma3     -D BATCH_TILES=8,16,24,32   -D DIV=DIV -o src/f32-vcos/gen/f32-vcos-fma3-rational-5-4-div.c &
tools/xngen src/f32-vsin/rational-5-4.c.in -D FUN=COS -D ARCH=avx512f  -D BATCH_TILES=16,32,48,64  -D DIV=DIV -o src/f32-vcos/gen/f32-vcos-avx512f-rational-5-4-div.c &
tools/xngen src/f32-vsin/rational-5-4.c.in -D FUN=COS -D ARCH=hvx      -D BATCH_TILES=32,64,96,128 -D DIV=DIV -o src/f32-vcos/gen/f32-vcos-hvx-rational-5-4-div.c &

tools/xngen src/f32-vsin/rational-5-4.c.in -D FUN=COS -D ARCH=neon     -D BATCH_TILES=4,8,12,16    -D DIV=NR  -o src/f32-vcos/gen/f32-vcos-neon-rational-5-4-nr.c &
tools/xngen src/f32-vsin/rational-5-4.c.in -D FUN=COS -D ARCH=avx512f  -D BATCH_TILES=16,32,48,64  -D DIV=NR  -o src/f32-vcos/gen/f32-vcos-avx512f-rational-5-4-nr.c &

wait
