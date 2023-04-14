#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-vcmul/neon.c.in -D BATCH_TILE=4  -o src/f32-vcmul/gen/f32-vcmul-neon-x4.c &
tools/xngen src/f32-vcmul/neon.c.in -D BATCH_TILE=8  -o src/f32-vcmul/gen/f32-vcmul-neon-x8.c &
tools/xngen src/f32-vcmul/neon.c.in -D BATCH_TILE=12 -o src/f32-vcmul/gen/f32-vcmul-neon-x12.c &
tools/xngen src/f32-vcmul/neon.c.in -D BATCH_TILE=16 -o src/f32-vcmul/gen/f32-vcmul-neon-x16.c &

################################### x86 SSE ###################################
tools/xngen src/f32-vcmul/sse.c.in -D BATCH_TILE=4  -o src/f32-vcmul/gen/f32-vcmul-sse-x4.c &
tools/xngen src/f32-vcmul/sse.c.in -D BATCH_TILE=8  -o src/f32-vcmul/gen/f32-vcmul-sse-x8.c &
tools/xngen src/f32-vcmul/sse.c.in -D BATCH_TILE=12 -o src/f32-vcmul/gen/f32-vcmul-sse-x12.c &
tools/xngen src/f32-vcmul/sse.c.in -D BATCH_TILE=16 -o src/f32-vcmul/gen/f32-vcmul-sse-x16.c &

################################## Wasm SIMD ##################################
tools/xngen src/f32-vcmul/wasmsimd.c.in -D BATCH_TILE=4  -o src/f32-vcmul/gen/f32-vcmul-wasmsimd-x4.c &
tools/xngen src/f32-vcmul/wasmsimd.c.in -D BATCH_TILE=8  -o src/f32-vcmul/gen/f32-vcmul-wasmsimd-x8.c &
tools/xngen src/f32-vcmul/wasmsimd.c.in -D BATCH_TILE=12 -o src/f32-vcmul/gen/f32-vcmul-wasmsimd-x12.c &
tools/xngen src/f32-vcmul/wasmsimd.c.in -D BATCH_TILE=16 -o src/f32-vcmul/gen/f32-vcmul-wasmsimd-x16.c &

#################################### Scalar ###################################
tools/xngen src/f32-vcmul/scalar.c.in -D BATCH_TILE=1 -o src/f32-vcmul/gen/f32-vcmul-scalar-x1.c &
tools/xngen src/f32-vcmul/scalar.c.in -D BATCH_TILE=2 -o src/f32-vcmul/gen/f32-vcmul-scalar-x2.c &
tools/xngen src/f32-vcmul/scalar.c.in -D BATCH_TILE=4 -o src/f32-vcmul/gen/f32-vcmul-scalar-x4.c &
tools/xngen src/f32-vcmul/scalar.c.in -D BATCH_TILE=8 -o src/f32-vcmul/gen/f32-vcmul-scalar-x8.c &

################################## Unit tests #################################
tools/generate-vbinary-test.py --tester VCMulMicrokernelTester --spec test/f32-vcmul.yaml --output test/f32-vcmul.cc &

wait
