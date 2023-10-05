#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/f32-f16-vcvt/neon.c.in -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neon-u8.c &
tools/xngen src/f32-f16-vcvt/neon.c.in -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neon-u16.c &
tools/xngen src/f32-f16-vcvt/neon.c.in -D BATCH_TILE=24 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neon-u24.c &
tools/xngen src/f32-f16-vcvt/neon.c.in -D BATCH_TILE=32 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neon-u32.c &

tools/xngen src/f32-f16-vcvt/neonfp16.c.in -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neonfp16-u8.c &
tools/xngen src/f32-f16-vcvt/neonfp16.c.in -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neonfp16-u16.c &

################################# x86 128-bit #################################
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse2-u8.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse2-u16.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=24 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse2-u24.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=32 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse2-u32.c &

tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse41-u8.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse41-u16.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=24 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse41-u24.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=32 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse41-u32.c &

tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-u8.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-u16.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=24 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-u24.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=32 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-u32.c &

################################# x86 256-bit #################################
tools/xngen src/f32-f16-vcvt/f16c.c.in -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-f16c-u8.c &
tools/xngen src/f32-f16-vcvt/f16c.c.in -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-f16c-u16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-f16-vcvt/avx512skx.c.in -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx512skx-u16.c &
tools/xngen src/f32-f16-vcvt/avx512skx.c.in -D BATCH_TILE=32 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx512skx-u32.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D RELAXED=0 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmsimd-u8.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D RELAXED=0 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmsimd-u16.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=24 -D RELAXED=0 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmsimd-u24.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D RELAXED=0 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmsimd-u32.c &

tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D RELAXED=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-u8.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D RELAXED=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-u16.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=24 -D RELAXED=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-u24.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D RELAXED=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-u32.c &

#################################### Scalar ###################################
tools/xngen src/f32-f16-vcvt/scalar-bitcast.c.in -D BATCH_TILE=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-bitcast-u1.c &
tools/xngen src/f32-f16-vcvt/scalar-bitcast.c.in -D BATCH_TILE=2 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-bitcast-u2.c &
tools/xngen src/f32-f16-vcvt/scalar-bitcast.c.in -D BATCH_TILE=3 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-bitcast-u3.c &
tools/xngen src/f32-f16-vcvt/scalar-bitcast.c.in -D BATCH_TILE=4 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-bitcast-u4.c &

tools/xngen src/f32-f16-vcvt/scalar-fabsf.c.in -D BATCH_TILE=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-fabsf-u1.c &
tools/xngen src/f32-f16-vcvt/scalar-fabsf.c.in -D BATCH_TILE=2 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-fabsf-u2.c &
tools/xngen src/f32-f16-vcvt/scalar-fabsf.c.in -D BATCH_TILE=3 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-fabsf-u3.c &
tools/xngen src/f32-f16-vcvt/scalar-fabsf.c.in -D BATCH_TILE=4 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-fabsf-u4.c &

wait
