#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/qs16-qs8-vcvt/neon.c.in -D BATCH_TILE=8  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-neon-x8.c &
tools/xngen src/qs16-qs8-vcvt/neon.c.in -D BATCH_TILE=16 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-neon-x16.c &
tools/xngen src/qs16-qs8-vcvt/neon.c.in -D BATCH_TILE=32 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-neon-x32.c &

################################## x86 SSE2 #################################
tools/xngen src/qs16-qs8-vcvt/sse2.c.in -D BATCH_TILE=4  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse2-x4.c &
tools/xngen src/qs16-qs8-vcvt/sse2.c.in -D BATCH_TILE=8  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse2-x8.c &
tools/xngen src/qs16-qs8-vcvt/sse2.c.in -D BATCH_TILE=16 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse2-x16.c &

################################## x86 SSSE3 #################################
tools/xngen src/qs16-qs8-vcvt/ssse3.c.in -D BATCH_TILE=4  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-ssse3-x4.c &
tools/xngen src/qs16-qs8-vcvt/ssse3.c.in -D BATCH_TILE=8  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-ssse3-x8.c &
tools/xngen src/qs16-qs8-vcvt/ssse3.c.in -D BATCH_TILE=16 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-ssse3-x16.c &

################################## x86 SSE4.1 #################################
tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=4  -D AVX=0 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse41-x4.c &
tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=0 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse41-x8.c &
tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=0 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-sse41-x16.c &

tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=4  -D AVX=1 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-avx-x4.c &
tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=1 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-avx-x8.c &
tools/xngen src/qs16-qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=1 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-avx-x16.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs16-qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-wasmsimd-x8.c &
tools/xngen src/qs16-qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-wasmsimd-x16.c &
tools/xngen src/qs16-qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-wasmsimd-x32.c &

#################################### Scalar ###################################
tools/xngen src/qs16-qs8-vcvt/scalar.c.in -D BATCH_TILE=1 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-scalar-x1.c &
tools/xngen src/qs16-qs8-vcvt/scalar.c.in -D BATCH_TILE=2 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-scalar-x2.c &
tools/xngen src/qs16-qs8-vcvt/scalar.c.in -D BATCH_TILE=4 -o src/qs16-qs8-vcvt/gen/qs16-qs8-vcvt-scalar-x4.c &

################################## Unit tests #################################
tools/generate-vcvt-test.py --spec test/qs16-qs8-vcvt.yaml --output test/qs16-qs8-vcvt.cc &

wait
