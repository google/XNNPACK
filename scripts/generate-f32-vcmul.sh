#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-vcmul/neon.c.in -D BATCH_TILE=4  -o src/f32-vcmul/gen/f32-vcmul-neon-u4.c &
tools/xngen src/f32-vcmul/neon.c.in -D BATCH_TILE=8  -o src/f32-vcmul/gen/f32-vcmul-neon-u8.c &
tools/xngen src/f32-vcmul/neon.c.in -D BATCH_TILE=12 -o src/f32-vcmul/gen/f32-vcmul-neon-u12.c &
tools/xngen src/f32-vcmul/neon.c.in -D BATCH_TILE=16 -o src/f32-vcmul/gen/f32-vcmul-neon-u16.c &

tools/xngen src/f16-vcmul/neonfp16arith.c.in -D BATCH_TILE=8  -o src/f16-vcmul/gen/f16-vcmul-neonfp16arith-u8.c &
tools/xngen src/f16-vcmul/neonfp16arith.c.in -D BATCH_TILE=16 -o src/f16-vcmul/gen/f16-vcmul-neonfp16arith-u16.c &
tools/xngen src/f16-vcmul/neonfp16arith.c.in -D BATCH_TILE=32 -o src/f16-vcmul/gen/f16-vcmul-neonfp16arith-u32.c &

################################### x86 SSE ###################################
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=0 -D BATCH_TILE=4  -o src/f32-vcmul/gen/f32-vcmul-sse-u4.c &
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=0 -D BATCH_TILE=8  -o src/f32-vcmul/gen/f32-vcmul-sse-u8.c &
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=0 -D BATCH_TILE=12 -o src/f32-vcmul/gen/f32-vcmul-sse-u12.c &
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=0 -D BATCH_TILE=16 -o src/f32-vcmul/gen/f32-vcmul-sse-u16.c &

################################### x86 FMA3 ###################################
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=1 -D BATCH_TILE=8 -o src/f32-vcmul/gen/f32-vcmul-fma3-u8.c &
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=1 -D BATCH_TILE=16 -o src/f32-vcmul/gen/f32-vcmul-fma3-u16.c &
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=1 -D BATCH_TILE=32 -o src/f32-vcmul/gen/f32-vcmul-fma3-u32.c &
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=1 -D BATCH_TILE=64 -o src/f32-vcmul/gen/f32-vcmul-fma3-u64.c &

################################### x86 AVX512SKX ###################################
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=512 -D BATCH_TILE=16 -o src/f32-vcmul/gen/f32-vcmul-avx512f-u16.c &
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=512 -D BATCH_TILE=32 -o src/f32-vcmul/gen/f32-vcmul-avx512f-u32.c &
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=512 -D BATCH_TILE=64 -o src/f32-vcmul/gen/f32-vcmul-avx512f-u64.c &
tools/xngen src/f32-vcmul/avx512f.c.in -D AVX=512 -D BATCH_TILE=128 -o src/f32-vcmul/gen/f32-vcmul-avx512f-u128.c &

################################## Wasm SIMD ##################################
tools/xngen src/f32-vcmul/wasmsimd.c.in -D BATCH_TILE=4  -o src/f32-vcmul/gen/f32-vcmul-wasmsimd-u4.c &
tools/xngen src/f32-vcmul/wasmsimd.c.in -D BATCH_TILE=8  -o src/f32-vcmul/gen/f32-vcmul-wasmsimd-u8.c &
tools/xngen src/f32-vcmul/wasmsimd.c.in -D BATCH_TILE=12 -o src/f32-vcmul/gen/f32-vcmul-wasmsimd-u12.c &
tools/xngen src/f32-vcmul/wasmsimd.c.in -D BATCH_TILE=16 -o src/f32-vcmul/gen/f32-vcmul-wasmsimd-u16.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vcmul/rvv.c.in -D LMUL=1 -o src/f32-vcmul/gen/f32-vcmul-rvv-u1v.c &
tools/xngen src/f32-vcmul/rvv.c.in -D LMUL=2 -o src/f32-vcmul/gen/f32-vcmul-rvv-u2v.c &
tools/xngen src/f32-vcmul/rvv.c.in -D LMUL=4 -o src/f32-vcmul/gen/f32-vcmul-rvv-u4v.c &

#################################### Scalar ###################################
tools/xngen src/f32-vcmul/scalar.c.in -D BATCH_TILE=1 -o src/f32-vcmul/gen/f32-vcmul-scalar-u1.c &
tools/xngen src/f32-vcmul/scalar.c.in -D BATCH_TILE=2 -o src/f32-vcmul/gen/f32-vcmul-scalar-u2.c &
tools/xngen src/f32-vcmul/scalar.c.in -D BATCH_TILE=4 -o src/f32-vcmul/gen/f32-vcmul-scalar-u4.c &
tools/xngen src/f32-vcmul/scalar.c.in -D BATCH_TILE=8 -o src/f32-vcmul/gen/f32-vcmul-scalar-u8.c &

wait
