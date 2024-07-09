#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-vgelu/scalar.c.in -D BATCH_TILES=1,2,4 -o src/f32-vgelu/gen/f32-vgelu-scalar.c &

##################################### SIMD #####################################
tools/xngen src/f32-vgelu/rational-12-10.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8 -D DIV=DIV -o src/f32-vgelu/gen/f32-vgelu-scalar-rational-12-10-div.c &
tools/xngen src/f32-vgelu/rational-12-10.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vgelu/gen/f32-vgelu-sse2-rational-12-10-div.c &
tools/xngen src/f32-vgelu/rational-12-10.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vgelu/gen/f32-vgelu-avx-rational-12-10-div.c &
tools/xngen src/f32-vgelu/rational-12-10.c.in -D ARCH=fma3 -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vgelu/gen/f32-vgelu-fma3-rational-12-10-div.c &
tools/xngen src/f32-vgelu/rational-12-10.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=DIV -o src/f32-vgelu/gen/f32-vgelu-avx512f-rational-12-10-div.c &
tools/xngen src/f32-vgelu/rational-12-10.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vgelu/gen/f32-vgelu-neon-rational-12-10-div.c &
tools/xngen src/f32-vgelu/rational-12-10.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vgelu/gen/f32-vgelu-wasmsimd-rational-12-10-div.c &
tools/xngen src/f32-vgelu/rational-12-10.c.in -D ARCH=hvx -D BATCH_TILES=32,64,128 -D DIV=DIV -o src/f32-vgelu/gen/f32-vgelu-hvx-rational-12-10-div.c &

tools/xngen src/f32-vgelu/rational-12-10.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=NR -o src/f32-vgelu/gen/f32-vgelu-avx512f-rational-12-10-nr.c &

wait
