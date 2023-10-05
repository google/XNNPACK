#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-prelu/scalar.c.in -D CHANNEL_TILE=1 -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-scalar-2x1.c &
tools/xngen src/f32-prelu/scalar.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-scalar-2x4.c &

##################################### WAsm ####################################
tools/xngen src/f32-prelu/wasm.c.in -D CHANNEL_TILE=1 -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-wasm-2x1.c &
tools/xngen src/f32-prelu/wasm.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-wasm-2x4.c &

################################### ARM NEON ##################################
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=4  -D ROW_TILE=1 -o src/f32-prelu/gen/f32-prelu-neon-1x4.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=8  -D ROW_TILE=1 -o src/f32-prelu/gen/f32-prelu-neon-1x8.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=16 -D ROW_TILE=1 -o src/f32-prelu/gen/f32-prelu-neon-1x16.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=4  -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-neon-2x4.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-neon-2x8.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-neon-2x16.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=4  -D ROW_TILE=4 -o src/f32-prelu/gen/f32-prelu-neon-4x4.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=8  -D ROW_TILE=4 -o src/f32-prelu/gen/f32-prelu-neon-4x8.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=16 -D ROW_TILE=4 -o src/f32-prelu/gen/f32-prelu-neon-4x16.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=4  -D ROW_TILE=1 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-laneselect-1x4.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=8  -D ROW_TILE=1 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-laneselect-1x8.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=16 -D ROW_TILE=1 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-laneselect-1x16.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=4  -D ROW_TILE=2 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-laneselect-2x4.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-laneselect-2x8.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-laneselect-2x16.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=4  -D ROW_TILE=4 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-laneselect-4x4.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=8  -D ROW_TILE=4 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-laneselect-4x8.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=16 -D ROW_TILE=4 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-laneselect-4x16.c &

tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=4  -D ROW_TILE=1 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-1x4.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=8  -D ROW_TILE=1 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-1x8.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=16 -D ROW_TILE=1 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-1x16.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=4  -D ROW_TILE=2 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-2x4.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-2x8.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-2x16.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=4  -D ROW_TILE=4 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-4x4.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=8  -D ROW_TILE=4 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-4x8.c &
tools/xngen src/f32-prelu/wasmsimd-laneselect.c.in -D CHANNEL_TILE=16 -D ROW_TILE=4 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-laneselect-4x16.c &

tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=4  -D ROW_TILE=1 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-iminmax-1x4.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=8  -D ROW_TILE=1 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-iminmax-1x8.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=16 -D ROW_TILE=1 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-iminmax-1x16.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=4  -D ROW_TILE=2 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-iminmax-2x4.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=8  -D ROW_TILE=2 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-iminmax-2x8.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=16 -D ROW_TILE=2 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-iminmax-2x16.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=4  -D ROW_TILE=4 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-iminmax-4x4.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=8  -D ROW_TILE=4 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-iminmax-4x8.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=16 -D ROW_TILE=4 -D RELAXED=0 -o src/f32-prelu/gen/f32-prelu-wasmsimd-iminmax-4x16.c &

tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=4  -D ROW_TILE=1 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-1x4.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=8  -D ROW_TILE=1 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-1x8.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=16 -D ROW_TILE=1 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-1x16.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=4  -D ROW_TILE=2 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-2x4.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=8  -D ROW_TILE=2 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-2x8.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=16 -D ROW_TILE=2 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-2x16.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=4  -D ROW_TILE=4 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-4x4.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=8  -D ROW_TILE=4 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-4x8.c &
tools/xngen src/f32-prelu/wasmsimd-iminmax.c.in    -D CHANNEL_TILE=16 -D ROW_TILE=4 -D RELAXED=1 -o src/f32-prelu/gen/f32-prelu-wasmrelaxedsimd-iminmax-4x16.c &

################################### ARM NEON ##################################
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-neon-2x4.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-neon-2x8.c &

############################# x86 SSE/SSE2/SSE4.1 #############################
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D SSE=1 -o src/f32-prelu/gen/f32-prelu-sse-2x4.c &
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -D SSE=1 -o src/f32-prelu/gen/f32-prelu-sse-2x8.c &
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D SSE=2 -o src/f32-prelu/gen/f32-prelu-sse2-2x4.c &
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -D SSE=2 -o src/f32-prelu/gen/f32-prelu-sse2-2x8.c &
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D SSE=4 -o src/f32-prelu/gen/f32-prelu-sse41-2x4.c &
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -D SSE=4 -o src/f32-prelu/gen/f32-prelu-sse41-2x8.c &

################################### x86 AVX ###################################
tools/xngen src/f32-prelu/avx.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-avx-2x8.c &
tools/xngen src/f32-prelu/avx.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-avx-2x16.c &

################################## x86 AVX512 #################################
tools/xngen src/f32-prelu/avx512f.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-avx512f-2x16.c &
tools/xngen src/f32-prelu/avx512f.c.in -D CHANNEL_TILE=32 -D ROW_TILE=2 -o src/f32-prelu/gen/f32-prelu-avx512f-2x32.c &

wait
