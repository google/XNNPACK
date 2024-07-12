#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-vlog/scalar-log.c.in -D BATCH_TILES=1,2,4 -o src/f32-vlog/gen/f32-vlog-scalar-log.c &

##################################### SIMD #####################################
tools/xngen src/f32-vlog/rational-3-3.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8 -D DIV=DIV -o src/f32-vlog/gen/f32-vlog-scalar-rational-3-3-div.c &
tools/xngen src/f32-vlog/rational-3-3.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vlog/gen/f32-vlog-sse2-rational-3-3-div.c &
tools/xngen src/f32-vlog/rational-3-3.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vlog/gen/f32-vlog-avx2-rational-3-3-div.c &
tools/xngen src/f32-vlog/rational-3-3.c.in -D ARCH=fma3 -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vlog/gen/f32-vlog-fma3-rational-3-3-div.c &
tools/xngen src/f32-vlog/rational-3-3.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=DIV -o src/f32-vlog/gen/f32-vlog-avx512f-rational-3-3-div.c &
tools/xngen src/f32-vlog/rational-3-3.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vlog/gen/f32-vlog-neon-rational-3-3-div.c &
tools/xngen src/f32-vlog/rational-3-3.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vlog/gen/f32-vlog-wasmsimd-rational-3-3-div.c &

tools/xngen src/f32-vlog/rational-3-3.c.in -D ARCH=fma3 -D BATCH_TILES=8,16,24,32 -D DIV=NR -o src/f32-vlog/gen/f32-vlog-fma3-rational-3-3-nr.c &
tools/xngen src/f32-vlog/rational-3-3.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=NR -o src/f32-vlog/gen/f32-vlog-avx512f-rational-3-3-nr.c &

wait
