#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/f32-f16-vcvt/neon.c.in -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neon-x8.c &
tools/xngen src/f32-f16-vcvt/neon.c.in -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neon-x16.c &
tools/xngen src/f32-f16-vcvt/neon.c.in -D BATCH_TILE=24 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neon-x24.c &
tools/xngen src/f32-f16-vcvt/neon.c.in -D BATCH_TILE=32 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neon-x32.c &

tools/xngen src/f32-f16-vcvt/neonfp16.c.in -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neonfp16-x8.c &
tools/xngen src/f32-f16-vcvt/neonfp16.c.in -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-neonfp16-x16.c &

################################# x86 128-bit #################################
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse2-x8.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse2-x16.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=24 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse2-x24.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=32 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse2-x32.c &

tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse41-x8.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse41-x16.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=24 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse41-x24.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=32 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-sse41-x32.c &

tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-x8.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-x16.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=24 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-x24.c &
tools/xngen src/f32-f16-vcvt/sse.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=32 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx-x32.c &

################################# x86 256-bit #################################
tools/xngen src/f32-f16-vcvt/f16c.c.in -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/f32-f16-vcvt-f16c-x8.c &
tools/xngen src/f32-f16-vcvt/f16c.c.in -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-f16c-x16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-f16-vcvt/avx512skx.c.in -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx512skx-x16.c &
tools/xngen src/f32-f16-vcvt/avx512skx.c.in -D BATCH_TILE=32 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-avx512skx-x32.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D RELAXED=0 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmsimd-x8.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D RELAXED=0 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmsimd-x16.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=24 -D RELAXED=0 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmsimd-x24.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D RELAXED=0 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmsimd-x32.c &

tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D RELAXED=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-x8.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D RELAXED=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-x16.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=24 -D RELAXED=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-x24.c &
tools/xngen src/f32-f16-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D RELAXED=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-wasmrelaxedsimd-x32.c &

#################################### Scalar ###################################
tools/xngen src/f32-f16-vcvt/scalar-bitcast.c.in -D BATCH_TILE=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-bitcast-x1.c &
tools/xngen src/f32-f16-vcvt/scalar-bitcast.c.in -D BATCH_TILE=2 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-bitcast-x2.c &
tools/xngen src/f32-f16-vcvt/scalar-bitcast.c.in -D BATCH_TILE=3 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-bitcast-x3.c &
tools/xngen src/f32-f16-vcvt/scalar-bitcast.c.in -D BATCH_TILE=4 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-bitcast-x4.c &

tools/xngen src/f32-f16-vcvt/scalar-fabsf.c.in -D BATCH_TILE=1 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-fabsf-x1.c &
tools/xngen src/f32-f16-vcvt/scalar-fabsf.c.in -D BATCH_TILE=2 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-fabsf-x2.c &
tools/xngen src/f32-f16-vcvt/scalar-fabsf.c.in -D BATCH_TILE=3 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-fabsf-x3.c &
tools/xngen src/f32-f16-vcvt/scalar-fabsf.c.in -D BATCH_TILE=4 -o src/f32-f16-vcvt/gen/f32-f16-vcvt-scalar-fabsf-x4.c &

################################## Unit tests #################################
tools/generate-vcvt-test.py --spec test/f32-f16-vcvt.yaml --output test/f32-f16-vcvt.cc &

wait
