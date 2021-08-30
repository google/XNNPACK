#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-vlrelu/scalar.c.in -D BATCH_TILE=1 -D WASM=0 -o src/f32-vlrelu/gen/vlrelu-scalar-x1.c &
tools/xngen src/f32-vlrelu/scalar.c.in -D BATCH_TILE=2 -D WASM=0 -o src/f32-vlrelu/gen/vlrelu-scalar-x2.c &
tools/xngen src/f32-vlrelu/scalar.c.in -D BATCH_TILE=4 -D WASM=0 -o src/f32-vlrelu/gen/vlrelu-scalar-x4.c &

##################################### WAsm ####################################
tools/xngen src/f32-vlrelu/wasm.c.in -D BATCH_TILE=1 -D WASM=1 -o src/f32-vlrelu/gen/vlrelu-wasm-x1.c &
tools/xngen src/f32-vlrelu/wasm.c.in -D BATCH_TILE=2 -D WASM=1 -o src/f32-vlrelu/gen/vlrelu-wasm-x2.c &
tools/xngen src/f32-vlrelu/wasm.c.in -D BATCH_TILE=4 -D WASM=1 -o src/f32-vlrelu/gen/vlrelu-wasm-x4.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vlrelu/wasmsimd-bitselect.c.in -D BATCH_TILE=4 -o src/f32-vlrelu/gen/vlrelu-wasmsimd-bitselect-x4.c &
tools/xngen src/f32-vlrelu/wasmsimd-bitselect.c.in -D BATCH_TILE=8 -o src/f32-vlrelu/gen/vlrelu-wasmsimd-bitselect-x8.c &

tools/xngen src/f32-vlrelu/wasmsimd-minmax.c.in -D BATCH_TILE=4 -o src/f32-vlrelu/gen/vlrelu-wasmsimd-minmax-x4.c &
tools/xngen src/f32-vlrelu/wasmsimd-minmax.c.in -D BATCH_TILE=8 -o src/f32-vlrelu/gen/vlrelu-wasmsimd-minmax-x8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vlrelu/neon.c.in -D BATCH_TILE=4 -o src/f32-vlrelu/gen/vlrelu-neon-x4.c &
tools/xngen src/f32-vlrelu/neon.c.in -D BATCH_TILE=8 -o src/f32-vlrelu/gen/vlrelu-neon-x8.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=4 -D SSE=1 -o src/f32-vlrelu/gen/vlrelu-sse-x4.c &
tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=8 -D SSE=1 -o src/f32-vlrelu/gen/vlrelu-sse-x8.c &

tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=4 -D SSE=2 -o src/f32-vlrelu/gen/vlrelu-sse2-x4.c &
tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=8 -D SSE=2 -o src/f32-vlrelu/gen/vlrelu-sse2-x8.c &

tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=4 -D SSE=4 -o src/f32-vlrelu/gen/vlrelu-sse41-x4.c &
tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=8 -D SSE=4 -o src/f32-vlrelu/gen/vlrelu-sse41-x8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vlrelu/avx.c.in -D BATCH_TILE=8  -o src/f32-vlrelu/gen/vlrelu-avx-x8.c &
tools/xngen src/f32-vlrelu/avx.c.in -D BATCH_TILE=16 -o src/f32-vlrelu/gen/vlrelu-avx-x16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vlrelu/avx512f.c.in -D BATCH_TILE=16 -o src/f32-vlrelu/gen/vlrelu-avx512f-x16.c &
tools/xngen src/f32-vlrelu/avx512f.c.in -D BATCH_TILE=32 -o src/f32-vlrelu/gen/vlrelu-avx512f-x32.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vlrelu.yaml --output test/f32-vlrelu.cc &

wait
