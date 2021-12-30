#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/f16-f32-vcvt/neon-int16.c.in -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-neon-int16-x8.c &
tools/xngen src/f16-f32-vcvt/neon-int16.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-neon-int16-x16.c &
tools/xngen src/f16-f32-vcvt/neon-int16.c.in -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/vcvt-neon-int16-x24.c &
tools/xngen src/f16-f32-vcvt/neon-int16.c.in -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-neon-int16-x32.c &

tools/xngen src/f16-f32-vcvt/neon-int32.c.in -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-neon-int32-x8.c &
tools/xngen src/f16-f32-vcvt/neon-int32.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-neon-int32-x16.c &
tools/xngen src/f16-f32-vcvt/neon-int32.c.in -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/vcvt-neon-int32-x24.c &
tools/xngen src/f16-f32-vcvt/neon-int32.c.in -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-neon-int32-x32.c &

tools/xngen src/f16-f32-vcvt/neonfp16.c.in -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-neonfp16-x8.c &
tools/xngen src/f16-f32-vcvt/neonfp16.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-neonfp16-x16.c &

################################# x86 128-bit #################################
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-sse2-int16-x8.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-sse2-int16-x16.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/vcvt-sse2-int16-x24.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-sse2-int16-x32.c &

tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-sse2-int32-x8.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-sse2-int32-x16.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/vcvt-sse2-int32-x24.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=2 -D AVX=0 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-sse2-int32-x32.c &

tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-sse41-int16-x8.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-sse41-int16-x16.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/vcvt-sse41-int16-x24.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-sse41-int16-x32.c &

tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-sse41-int32-x8.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-sse41-int32-x16.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/vcvt-sse41-int32-x24.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=0 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-sse41-int32-x32.c &

tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-avx-int16-x8.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-avx-int16-x16.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/vcvt-avx-int16-x24.c &
tools/xngen src/f16-f32-vcvt/sse-int16.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-avx-int16-x32.c &

tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-avx-int32-x8.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-avx-int32-x16.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/vcvt-avx-int32-x24.c &
tools/xngen src/f16-f32-vcvt/sse-int32.c.in -D SSE=4 -D AVX=1 -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-avx-int32-x32.c &

################################# x86 256-bit #################################
tools/xngen src/f16-f32-vcvt/f16c.c.in -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-f16c-x8.c &
tools/xngen src/f16-f32-vcvt/f16c.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-f16c-x16.c &

################################# x86 512-bit #################################
tools/xngen src/f16-f32-vcvt/avx512skx.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-avx512skx-x16.c &
tools/xngen src/f16-f32-vcvt/avx512skx.c.in -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-avx512skx-x32.c &

################################## WAsm SIMD ##################################
tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-wasmsimd-int16-x8.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-wasmsimd-int16-x16.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/vcvt-wasmsimd-int16-x24.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int16.c.in -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-wasmsimd-int16-x32.c &

tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-wasmsimd-int32-x8.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-wasmsimd-int32-x16.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=24 -o src/f16-f32-vcvt/gen/vcvt-wasmsimd-int32-x24.c &
tools/xngen src/f16-f32-vcvt/wasmsimd-int32.c.in -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-wasmsimd-int32-x32.c &

#################################### Scalar ###################################
tools/xngen src/f16-f32-vcvt/scalar.c.in -D BATCH_TILE=1 -o src/f16-f32-vcvt/gen/vcvt-scalar-x1.c &
tools/xngen src/f16-f32-vcvt/scalar.c.in -D BATCH_TILE=2 -o src/f16-f32-vcvt/gen/vcvt-scalar-x2.c &
tools/xngen src/f16-f32-vcvt/scalar.c.in -D BATCH_TILE=3 -o src/f16-f32-vcvt/gen/vcvt-scalar-x3.c &
tools/xngen src/f16-f32-vcvt/scalar.c.in -D BATCH_TILE=4 -o src/f16-f32-vcvt/gen/vcvt-scalar-x4.c &

################################## Unit tests #################################
tools/generate-vcvt-test.py --spec test/f16-f32-vcvt.yaml --output test/f16-f32-vcvt.cc &

wait
