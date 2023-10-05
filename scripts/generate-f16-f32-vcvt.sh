#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/f16-f32-vcvt/neon-int16.c.in -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/f16-f32-vcvt-neon-int16-u8.c &
tools/xngen src/f16-f32-vcvt/neon-int16.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-neon-int16-u16.c &
tools/xngen src/f16-f32-vcvt/neon-int16.c.in -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-neon-int16-u24.c &
tools/xngen src/f16-f32-vcvt/neon-int16.c.in -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-neon-int16-u32.c &

tools/xngen src/f16-f32-vcvt/neon-int32.c.in -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/f16-f32-vcvt-neon-int32-u8.c &
tools/xngen src/f16-f32-vcvt/neon-int32.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-neon-int32-u16.c &
tools/xngen src/f16-f32-vcvt/neon-int32.c.in -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-neon-int32-u24.c &
tools/xngen src/f16-f32-vcvt/neon-int32.c.in -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-neon-int32-u32.c &

tools/xngen src/f16-f32-vcvt/neonfp16.c.in -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/f16-f32-vcvt-neonfp16-u8.c &
tools/xngen src/f16-f32-vcvt/neonfp16.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-neonfp16-u16.c &

################################# x86 128-bit #################################
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse2-int16-u8.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse2-int16-u16.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse2-int16-u24.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse2-int16-u32.c &

tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse2-int32-u8.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse2-int32-u16.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse2-int32-u24.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse2-int32-u32.c &

tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse41-int16-u8.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse41-int16-u16.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse41-int16-u24.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse41-int16-u32.c &

tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse41-int32-u8.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse41-int32-u16.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse41-int32-u24.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-sse41-int32-u32.c &

tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int16-u8.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int16-u16.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int16-u24.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int16-u32.c &

tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int32-u8.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int32-u16.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int32-u24.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-avx-int32-u32.c &

################################# x86 256-bit #################################
tools/xngen src/f16-f32-vcvt/f16c.c.in -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/f16-f32-vcvt-f16c-u8.c &
tools/xngen src/f16-f32-vcvt/f16c.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-f16c-u16.c &

################################# x86 512-bit #################################
tools/xngen src/f16-f32-vcvt/avx512skx.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-avx512skx-u16.c &
tools/xngen src/f16-f32-vcvt/avx512skx.c.in -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-avx512skx-u32.c &

################################## WAsm SIMD ##################################
tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=8  -D RELAXED=0 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmsimd-int16-u8.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=16 -D RELAXED=0 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmsimd-int16-u16.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=24 -D RELAXED=0 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmsimd-int16-u24.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=32 -D RELAXED=0 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmsimd-int16-u32.c &

tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=8  -D RELAXED=1 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int16-u8.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=16 -D RELAXED=1 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int16-u16.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=24 -D RELAXED=1 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int16-u24.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=32 -D RELAXED=1 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int16-u32.c &

tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=8  -D RELAXED=0 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmsimd-int32-u8.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=16 -D RELAXED=0 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmsimd-int32-u16.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=24 -D RELAXED=0 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmsimd-int32-u24.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=32 -D RELAXED=0 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmsimd-int32-u32.c &

tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=8  -D RELAXED=1 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int32-u8.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=16 -D RELAXED=1 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int32-u16.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=24 -D RELAXED=1 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int32-u24.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=32 -D RELAXED=1 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-wasmrelaxedsimd-int32-u32.c &

#################################### Scalar ###################################
tools/xngen src/f16-f32-vcvt/scalar.c.in -D BATCH_TILE=1 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-scalar-u1.c &
tools/xngen src/f16-f32-vcvt/scalar.c.in -D BATCH_TILE=2 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-scalar-u2.c &
tools/xngen src/f16-f32-vcvt/scalar.c.in -D BATCH_TILE=3 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-scalar-u3.c &
tools/xngen src/f16-f32-vcvt/scalar.c.in -D BATCH_TILE=4 -o src/f16-f32-vcvt/gen/f16-f32-vcvt-scalar-u4.c &

wait
