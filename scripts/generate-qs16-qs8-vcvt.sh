#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/qs16-qs8-vcvt/neon.c.in -D BATCH_TILE=8  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-neon-u8.c &
tools/xngen src/qs16-qs8-vcvt/neon.c.in -D BATCH_TILE=16 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-neon-u16.c &
tools/xngen src/qs16-qs8-vcvt/neon.c.in -D BATCH_TILE=32 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-neon-u32.c &

################################## x86 SSE2 #################################
tools/xngen src/qs16-qs8-vcvt/sse2.c.in -D BATCH_TILE=4  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse2-u4.c &
tools/xngen src/qs16-qs8-vcvt/sse2.c.in -D BATCH_TILE=8  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse2-u8.c &
tools/xngen src/qs16-qs8-vcvt/sse2.c.in -D BATCH_TILE=16 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse2-u16.c &

################################## x86 SSSE3 #################################
tools/xngen src/qs16-qs8-vcvt/ssse3.c.in -D BATCH_TILE=4  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-ssse3-u4.c &
tools/xngen src/qs16-qs8-vcvt/ssse3.c.in -D BATCH_TILE=8  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-ssse3-u8.c &
tools/xngen src/qs16-qs8-vcvt/ssse3.c.in -D BATCH_TILE=16 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-ssse3-u16.c &

################################## x86 SSE4.1 #################################
tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=4  -D AVX=0 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse41-u4.c &
tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=0 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse41-u8.c &
tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=0 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse41-u16.c &

tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=4  -D AVX=1 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-avx-u4.c &
tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=1 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-avx-u8.c &
tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=1 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-avx-u16.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs16-qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-wasmsimd-u8.c &
tools/xngen src/qs16-qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-wasmsimd-u16.c &
tools/xngen src/qs16-qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-wasmsimd-u32.c &

#################################### Scalar ###################################
tools/xngen src/qs16-qs8-vcvt/scalar.c.in -D BATCH_TILE=1 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-scalar-u1.c &
tools/xngen src/qs16-qs8-vcvt/scalar.c.in -D BATCH_TILE=2 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-scalar-u2.c &
tools/xngen src/qs16-qs8-vcvt/scalar.c.in -D BATCH_TILE=4 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-scalar-u4.c &

wait
