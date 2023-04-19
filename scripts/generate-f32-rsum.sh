#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-rsum/neon.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-neon-x4.c &
tools/xngen src/f32-rsum/neon.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-neon-x8-acc2.c &
tools/xngen src/f32-rsum/neon.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-neon-x12-acc3.c &
tools/xngen src/f32-rsum/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-neon-x16-acc2.c &
tools/xngen src/f32-rsum/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-neon-x16-acc4.c &

################################### x86 SSE ###################################
tools/xngen src/f32-rsum/sse.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-sse-x4.c &
tools/xngen src/f32-rsum/sse.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-sse-x8-acc2.c &
tools/xngen src/f32-rsum/sse.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-sse-x12-acc3.c &
tools/xngen src/f32-rsum/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-sse-x16-acc2.c &
tools/xngen src/f32-rsum/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-sse-x16-acc4.c &

################################### x86 AVX ###################################
tools/xngen src/f32-rsum/avx.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-avx-x8.c &
tools/xngen src/f32-rsum/avx.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-avx-x16-acc2.c &
tools/xngen src/f32-rsum/avx.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-avx-x24-acc3.c &
tools/xngen src/f32-rsum/avx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-avx-x32-acc2.c &
tools/xngen src/f32-rsum/avx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-avx-x32-acc4.c &

################################## Wasm SIMD ##################################
tools/xngen src/f32-rsum/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-wasmsimd-x4.c &
tools/xngen src/f32-rsum/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-wasmsimd-x8-acc2.c &
tools/xngen src/f32-rsum/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-wasmsimd-x12-acc3.c &
tools/xngen src/f32-rsum/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-wasmsimd-x16-acc2.c &
tools/xngen src/f32-rsum/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-wasmsimd-x16-acc4.c &

#################################### Scalar ###################################
tools/xngen src/f32-rsum/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-scalar-x1.c &
tools/xngen src/f32-rsum/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-scalar-x2-acc2.c &
tools/xngen src/f32-rsum/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-scalar-x3-acc3.c &
tools/xngen src/f32-rsum/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-scalar-x4-acc2.c &
tools/xngen src/f32-rsum/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-scalar-x4-acc4.c &

################################## Unit tests #################################
tools/generate-reduce-test.py --tester RSumMicrokernelTester --spec test/f32-rsum.yaml --output test/f32-rsum.cc &

wait
