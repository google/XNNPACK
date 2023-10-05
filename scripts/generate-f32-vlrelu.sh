#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-vlrelu/scalar.c.in -D BATCH_TILE=1 -D WASM=0 -o src/f32-vlrelu/gen/f32-vlrelu-scalar-u1.c &
tools/xngen src/f32-vlrelu/scalar.c.in -D BATCH_TILE=2 -D WASM=0 -o src/f32-vlrelu/gen/f32-vlrelu-scalar-u2.c &
tools/xngen src/f32-vlrelu/scalar.c.in -D BATCH_TILE=4 -D WASM=0 -o src/f32-vlrelu/gen/f32-vlrelu-scalar-u4.c &

##################################### WAsm ####################################
tools/xngen src/f32-vlrelu/wasm.c.in -D BATCH_TILE=1 -D WASM=1 -o src/f32-vlrelu/gen/f32-vlrelu-wasm-u1.c &
tools/xngen src/f32-vlrelu/wasm.c.in -D BATCH_TILE=2 -D WASM=1 -o src/f32-vlrelu/gen/f32-vlrelu-wasm-u2.c &
tools/xngen src/f32-vlrelu/wasm.c.in -D BATCH_TILE=4 -D WASM=1 -o src/f32-vlrelu/gen/f32-vlrelu-wasm-u4.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vlrelu/wasmsimd-laneselect.c.in -D BATCH_TILE=4 -D RELAXED=0 -o src/f32-vlrelu/gen/f32-vlrelu-wasmsimd-laneselect-u4.c &
tools/xngen src/f32-vlrelu/wasmsimd-laneselect.c.in -D BATCH_TILE=8 -D RELAXED=0 -o src/f32-vlrelu/gen/f32-vlrelu-wasmsimd-laneselect-u8.c &

tools/xngen src/f32-vlrelu/wasmsimd-laneselect.c.in -D BATCH_TILE=4 -D RELAXED=1 -o src/f32-vlrelu/gen/f32-vlrelu-wasmrelaxedsimd-laneselect-u4.c &
tools/xngen src/f32-vlrelu/wasmsimd-laneselect.c.in -D BATCH_TILE=8 -D RELAXED=1 -o src/f32-vlrelu/gen/f32-vlrelu-wasmrelaxedsimd-laneselect-u8.c &

tools/xngen src/f32-vlrelu/wasmsimd-iminmax.c.in    -D BATCH_TILE=4 -D RELAXED=0 -o src/f32-vlrelu/gen/f32-vlrelu-wasmsimd-iminmax-u4.c &
tools/xngen src/f32-vlrelu/wasmsimd-iminmax.c.in    -D BATCH_TILE=8 -D RELAXED=0 -o src/f32-vlrelu/gen/f32-vlrelu-wasmsimd-iminmax-u8.c &

tools/xngen src/f32-vlrelu/wasmsimd-iminmax.c.in    -D BATCH_TILE=4 -D RELAXED=1 -o src/f32-vlrelu/gen/f32-vlrelu-wasmrelaxedsimd-iminmax-u4.c &
tools/xngen src/f32-vlrelu/wasmsimd-iminmax.c.in    -D BATCH_TILE=8 -D RELAXED=1 -o src/f32-vlrelu/gen/f32-vlrelu-wasmrelaxedsimd-iminmax-u8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vlrelu/neon.c.in -D BATCH_TILE=4 -o src/f32-vlrelu/gen/f32-vlrelu-neon-u4.c &
tools/xngen src/f32-vlrelu/neon.c.in -D BATCH_TILE=8 -o src/f32-vlrelu/gen/f32-vlrelu-neon-u8.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=4 -D SSE=1 -o src/f32-vlrelu/gen/f32-vlrelu-sse-u4.c &
tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=8 -D SSE=1 -o src/f32-vlrelu/gen/f32-vlrelu-sse-u8.c &

tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=4 -D SSE=2 -o src/f32-vlrelu/gen/f32-vlrelu-sse2-u4.c &
tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=8 -D SSE=2 -o src/f32-vlrelu/gen/f32-vlrelu-sse2-u8.c &

tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=4 -D SSE=4 -o src/f32-vlrelu/gen/f32-vlrelu-sse41-u4.c &
tools/xngen src/f32-vlrelu/sse.c.in -D BATCH_TILE=8 -D SSE=4 -o src/f32-vlrelu/gen/f32-vlrelu-sse41-u8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vlrelu/avx.c.in -D BATCH_TILE=8  -o src/f32-vlrelu/gen/f32-vlrelu-avx-u8.c &
tools/xngen src/f32-vlrelu/avx.c.in -D BATCH_TILE=16 -o src/f32-vlrelu/gen/f32-vlrelu-avx-u16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vlrelu/avx512f.c.in -D BATCH_TILE=16 -o src/f32-vlrelu/gen/f32-vlrelu-avx512f-u16.c &
tools/xngen src/f32-vlrelu/avx512f.c.in -D BATCH_TILE=32 -o src/f32-vlrelu/gen/f32-vlrelu-avx512f-u32.c &

wait
