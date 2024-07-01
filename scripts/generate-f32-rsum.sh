#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-rsum/neon.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-neon-u4.c &
tools/xngen src/f32-rsum/neon.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-neon-u8-acc2.c &
tools/xngen src/f32-rsum/neon.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-neon-u12-acc3.c &
tools/xngen src/f32-rsum/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-neon-u16-acc2.c &
tools/xngen src/f32-rsum/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-neon-u16-acc4.c &

################################### x86 SSE ###################################
tools/xngen src/f32-rsum/sse.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-sse-u4.c &
tools/xngen src/f32-rsum/sse.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-sse-u8-acc2.c &
tools/xngen src/f32-rsum/sse.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-sse-u12-acc3.c &
tools/xngen src/f32-rsum/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-sse-u16-acc2.c &
tools/xngen src/f32-rsum/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-sse-u16-acc4.c &

################################### x86 AVX ###################################
tools/xngen src/f32-rsum/avx.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-avx-u8.c &
tools/xngen src/f32-rsum/avx.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-avx-u16-acc2.c &
tools/xngen src/f32-rsum/avx.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-avx-u24-acc3.c &
tools/xngen src/f32-rsum/avx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-avx-u32-acc2.c &
tools/xngen src/f32-rsum/avx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-avx-u32-acc4.c &

################################## x86 AVX512 #################################
tools/xngen src/f32-rsum/avx512f.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-avx512f-u16.c &
tools/xngen src/f32-rsum/avx512f.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-avx512f-u32-acc2.c &
tools/xngen src/f32-rsum/avx512f.c.in -D BATCH_TILE=48 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-avx512f-u48-acc3.c &
tools/xngen src/f32-rsum/avx512f.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-avx512f-u64-acc2.c &
tools/xngen src/f32-rsum/avx512f.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-avx512f-u64-acc4.c &

################################## Hexagon HVX ################################
tools/xngen src/f32-rsum/hvx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-hvx-u32.c &
tools/xngen src/f32-rsum/hvx.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-hvx-u64-acc2.c &
tools/xngen src/f32-rsum/hvx.c.in -D BATCH_TILE=96 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-hvx-u96-acc3.c &
tools/xngen src/f32-rsum/hvx.c.in -D BATCH_TILE=128 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-hvx-u128-acc2.c &
tools/xngen src/f32-rsum/hvx.c.in -D BATCH_TILE=128 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-hvx-u128-acc4.c &

################################## Wasm SIMD ##################################
tools/xngen src/f32-rsum/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-wasmsimd-u4.c &
tools/xngen src/f32-rsum/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-wasmsimd-u8-acc2.c &
tools/xngen src/f32-rsum/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-wasmsimd-u12-acc3.c &
tools/xngen src/f32-rsum/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-wasmsimd-u16-acc2.c &
tools/xngen src/f32-rsum/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-wasmsimd-u16-acc4.c &

#################################### Scalar ###################################
tools/xngen src/f32-rsum/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -o src/f32-rsum/gen/f32-rsum-scalar-u1.c &
tools/xngen src/f32-rsum/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-scalar-u2-acc2.c &
tools/xngen src/f32-rsum/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -o src/f32-rsum/gen/f32-rsum-scalar-u3-acc3.c &
tools/xngen src/f32-rsum/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -o src/f32-rsum/gen/f32-rsum-scalar-u4-acc2.c &
tools/xngen src/f32-rsum/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -o src/f32-rsum/gen/f32-rsum-scalar-u4-acc4.c &

wait
