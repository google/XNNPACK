#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-vexp/scalar-exp.c.in -D BATCH_TILES=1,2,4 -o src/f32-vexp/gen/f32-vexp-scalar-exp.c &

################################# SIMD wrappers ################################
tools/xngen src/f32-vexp/rational-3-2.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8 -D DIV=DIV -o src/f32-vexp/gen/f32-vexp-scalar-rational-3-2-div.c &
tools/xngen src/f32-vexp/rational-3-2.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vexp/gen/f32-vexp-sse2-rational-3-2-div.c &
tools/xngen src/f32-vexp/rational-3-2.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vexp/gen/f32-vexp-avx-rational-3-2-div.c &
tools/xngen src/f32-vexp/rational-3-2.c.in -D ARCH=fma3 -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vexp/gen/f32-vexp-fma3-rational-3-2-div.c &
tools/xngen src/f32-vexp/rational-3-2.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=DIV -o src/f32-vexp/gen/f32-vexp-avx512f-rational-3-2-div.c &
tools/xngen src/f32-vexp/rational-3-2.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vexp/gen/f32-vexp-neon-rational-3-2-div.c &
tools/xngen src/f32-vexp/rational-3-2.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vexp/gen/f32-vexp-wasmsimd-rational-3-2-div.c &
tools/xngen src/f32-vexp/rational-3-2.c.in -D ARCH=hvx -D BATCH_TILES=32,64,128 -D DIV=DIV -o src/f32-vexp/gen/f32-vexp-hvx-rational-3-2-div.c &

tools/xngen src/f32-vexp/rational-3-2.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=NR -o src/f32-vexp/gen/f32-vexp-avx512f-rational-3-2-nr.c &

wait
