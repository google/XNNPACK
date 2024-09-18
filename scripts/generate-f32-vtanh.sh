#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-scalar-rational-9-8-div.c &
tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-sse2-rational-9-8-div.c &
tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-rational-9-8-div.c &
tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-neon-rational-9-8-div.c &
tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-avx-rational-9-8-div.c &
tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=fma3 -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-fma3-rational-9-8-div.c &
tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-avx512f-rational-9-8-div.c &

tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16 -D DIV=NR -o src/f32-vtanh/gen/f32-vtanh-sse2-rational-9-8-nr.c &
tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16 -D DIV=NR -o src/f32-vtanh/gen/f32-vtanh-neon-rational-9-8-nr.c &
tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32 -D DIV=NR -o src/f32-vtanh/gen/f32-vtanh-avx-rational-9-8-nr.c &
tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=fma3 -D BATCH_TILES=8,16,24,32 -D DIV=NR -o src/f32-vtanh/gen/f32-vtanh-fma3-rational-9-8-nr.c &
tools/xngen src/f32-vtanh/rational-9-8.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=NR -o src/f32-vtanh/gen/f32-vtanh-avx512f-rational-9-8-nr.c &

wait
