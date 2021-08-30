#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-prelu/scalar.c.in -D CHANNEL_TILE=1 -D ROW_TILE=2 -o src/f32-prelu/gen/scalar-2x1.c &
tools/xngen src/f32-prelu/scalar.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -o src/f32-prelu/gen/scalar-2x4.c &

##################################### WAsm ####################################
tools/xngen src/f32-prelu/wasm.c.in -D CHANNEL_TILE=1 -D ROW_TILE=2 -o src/f32-prelu/gen/wasm-2x1.c &
tools/xngen src/f32-prelu/wasm.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -o src/f32-prelu/gen/wasm-2x4.c &

################################### ARM NEON ##################################
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=4  -D ROW_TILE=1 -o src/f32-prelu/gen/neon-1x4.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=8  -D ROW_TILE=1 -o src/f32-prelu/gen/neon-1x8.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=16 -D ROW_TILE=1 -o src/f32-prelu/gen/neon-1x16.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=4  -D ROW_TILE=2 -o src/f32-prelu/gen/neon-2x4.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -o src/f32-prelu/gen/neon-2x8.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f32-prelu/gen/neon-2x16.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=4  -D ROW_TILE=4 -o src/f32-prelu/gen/neon-4x4.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=8  -D ROW_TILE=4 -o src/f32-prelu/gen/neon-4x8.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=16 -D ROW_TILE=4 -o src/f32-prelu/gen/neon-4x16.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-prelu/wasmsimd-bitselect.c.in -D CHANNEL_TILE=4  -D ROW_TILE=1 -o src/f32-prelu/gen/wasmsimd-bitselect-1x4.c &
tools/xngen src/f32-prelu/wasmsimd-bitselect.c.in -D CHANNEL_TILE=8  -D ROW_TILE=1 -o src/f32-prelu/gen/wasmsimd-bitselect-1x8.c &
tools/xngen src/f32-prelu/wasmsimd-bitselect.c.in -D CHANNEL_TILE=16 -D ROW_TILE=1 -o src/f32-prelu/gen/wasmsimd-bitselect-1x16.c &
tools/xngen src/f32-prelu/wasmsimd-bitselect.c.in -D CHANNEL_TILE=4  -D ROW_TILE=2 -o src/f32-prelu/gen/wasmsimd-bitselect-2x4.c &
tools/xngen src/f32-prelu/wasmsimd-bitselect.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -o src/f32-prelu/gen/wasmsimd-bitselect-2x8.c &
tools/xngen src/f32-prelu/wasmsimd-bitselect.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f32-prelu/gen/wasmsimd-bitselect-2x16.c &
tools/xngen src/f32-prelu/wasmsimd-bitselect.c.in -D CHANNEL_TILE=4  -D ROW_TILE=4 -o src/f32-prelu/gen/wasmsimd-bitselect-4x4.c &
tools/xngen src/f32-prelu/wasmsimd-bitselect.c.in -D CHANNEL_TILE=8  -D ROW_TILE=4 -o src/f32-prelu/gen/wasmsimd-bitselect-4x8.c &
tools/xngen src/f32-prelu/wasmsimd-bitselect.c.in -D CHANNEL_TILE=16 -D ROW_TILE=4 -o src/f32-prelu/gen/wasmsimd-bitselect-4x16.c &

tools/xngen src/f32-prelu/wasmsimd-minmax.c.in    -D CHANNEL_TILE=4  -D ROW_TILE=1 -o src/f32-prelu/gen/wasmsimd-minmax-1x4.c &
tools/xngen src/f32-prelu/wasmsimd-minmax.c.in    -D CHANNEL_TILE=8  -D ROW_TILE=1 -o src/f32-prelu/gen/wasmsimd-minmax-1x8.c &
tools/xngen src/f32-prelu/wasmsimd-minmax.c.in    -D CHANNEL_TILE=16 -D ROW_TILE=1 -o src/f32-prelu/gen/wasmsimd-minmax-1x16.c &
tools/xngen src/f32-prelu/wasmsimd-minmax.c.in    -D CHANNEL_TILE=4  -D ROW_TILE=2 -o src/f32-prelu/gen/wasmsimd-minmax-2x4.c &
tools/xngen src/f32-prelu/wasmsimd-minmax.c.in    -D CHANNEL_TILE=8  -D ROW_TILE=2 -o src/f32-prelu/gen/wasmsimd-minmax-2x8.c &
tools/xngen src/f32-prelu/wasmsimd-minmax.c.in    -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f32-prelu/gen/wasmsimd-minmax-2x16.c &
tools/xngen src/f32-prelu/wasmsimd-minmax.c.in    -D CHANNEL_TILE=4  -D ROW_TILE=4 -o src/f32-prelu/gen/wasmsimd-minmax-4x4.c &
tools/xngen src/f32-prelu/wasmsimd-minmax.c.in    -D CHANNEL_TILE=8  -D ROW_TILE=4 -o src/f32-prelu/gen/wasmsimd-minmax-4x8.c &
tools/xngen src/f32-prelu/wasmsimd-minmax.c.in    -D CHANNEL_TILE=16 -D ROW_TILE=4 -o src/f32-prelu/gen/wasmsimd-minmax-4x16.c &

################################### ARM NEON ##################################
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -o src/f32-prelu/gen/neon-2x4.c &
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -o src/f32-prelu/gen/neon-2x8.c &

############################# x86 SSE/SSE2/SSE4.1 #############################
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D SSE=1 -o src/f32-prelu/gen/sse-2x4.c &
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -D SSE=1 -o src/f32-prelu/gen/sse-2x8.c &
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D SSE=2 -o src/f32-prelu/gen/sse2-2x4.c &
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -D SSE=2 -o src/f32-prelu/gen/sse2-2x8.c &
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D SSE=4 -o src/f32-prelu/gen/sse41-2x4.c &
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -D SSE=4 -o src/f32-prelu/gen/sse41-2x8.c &

################################### x86 AVX ###################################
tools/xngen src/f32-prelu/avx.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -o src/f32-prelu/gen/avx-2x8.c &
tools/xngen src/f32-prelu/avx.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f32-prelu/gen/avx-2x16.c &

################################## x86 AVX512 #################################
tools/xngen src/f32-prelu/avx512f.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f32-prelu/gen/avx512f-2x16.c &
tools/xngen src/f32-prelu/avx512f.c.in -D CHANNEL_TILE=32 -D ROW_TILE=2 -o src/f32-prelu/gen/avx512f-2x32.c &

################################## Unit tests #################################
tools/generate-prelu-test.py --spec test/f32-prelu.yaml --output test/f32-prelu.cc &

wait
