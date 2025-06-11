#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-vapproxgelu/scalar.c.in -D BATCH_TILES=1,2,4 -o src/f32-vapproxgelu/gen/f32-vapproxgelu-scalar.c &

##################################### SIMD #####################################
tools/xngen src/f32-vapproxgelu/rational-12-10.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8 -D DIV=DIV -o src/f32-vapproxgelu/gen/f32-vapproxgelu-scalar-rational-12-10-div.c &
tools/xngen src/f32-vapproxgelu/rational-12-10.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vapproxgelu/gen/f32-vapproxgelu-sse2-rational-12-10-div.c &
tools/xngen src/f32-vapproxgelu/rational-12-10.c.in -D ARCH=sse2fma -D BATCH_TILES=4,8 -D DIV=DIV -o src/f32-vapproxgelu/gen/f32-vapproxgelu-sse2fma-rational-12-10-div.c &
tools/xngen src/f32-vapproxgelu/rational-12-10.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vapproxgelu/gen/f32-vapproxgelu-avx-rational-12-10-div.c &
tools/xngen src/f32-vapproxgelu/rational-12-10.c.in -D ARCH=fma3 -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vapproxgelu/gen/f32-vapproxgelu-fma3-rational-12-10-div.c &
tools/xngen src/f32-vapproxgelu/rational-12-10.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=DIV -o src/f32-vapproxgelu/gen/f32-vapproxgelu-avx512f-rational-12-10-div.c &
tools/xngen src/f32-vapproxgelu/rational-12-10.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vapproxgelu/gen/f32-vapproxgelu-neon-rational-12-10-div.c &
tools/xngen src/f32-vapproxgelu/rational-12-10.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vapproxgelu/gen/f32-vapproxgelu-wasmsimd-rational-12-10-div.c &
tools/xngen src/f32-vapproxgelu/rational-12-10.c.in -D ARCH=hvx -D BATCH_TILES=32,64,128 -D DIV=DIV -o src/f32-vapproxgelu/gen/f32-vapproxgelu-hvx-rational-12-10-div.c &

tools/xngen src/f32-vapproxgelu/rational-12-10.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=NR -o src/f32-vapproxgelu/gen/f32-vapproxgelu-avx512f-rational-12-10-nr.c &

wait
