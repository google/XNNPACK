#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-vsqrt/scalar-sqrt.c.in -D BATCH_TILE=1 -o src/f32-vsqrt/gen/f32-vsqrt-scalar-sqrt-u1.c &
tools/xngen src/f32-vsqrt/scalar-sqrt.c.in -D BATCH_TILE=2 -o src/f32-vsqrt/gen/f32-vsqrt-scalar-sqrt-u2.c &
tools/xngen src/f32-vsqrt/scalar-sqrt.c.in -D BATCH_TILE=4 -o src/f32-vsqrt/gen/f32-vsqrt-scalar-sqrt-u4.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vsqrt/wasmsimd-sqrt.c.in -D BATCH_TILE=4 -o src/f32-vsqrt/gen/f32-vsqrt-wasmsimd-sqrt-u4.c &
tools/xngen src/f32-vsqrt/wasmsimd-sqrt.c.in -D BATCH_TILE=8 -o src/f32-vsqrt/gen/f32-vsqrt-wasmsimd-sqrt-u8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vsqrt/neon-sqrt.c.in -D BATCH_TILE=4 -o src/f32-vsqrt/gen/f32-vsqrt-aarch64-neon-sqrt-u4.c &
tools/xngen src/f32-vsqrt/neon-sqrt.c.in -D BATCH_TILE=8 -o src/f32-vsqrt/gen/f32-vsqrt-aarch64-neon-sqrt-u8.c &

tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=4  -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-u4.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-u8.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=12 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-u12.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-u16.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=20 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-u20.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=24 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-u24.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=28 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-u28.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=32 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-u32.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=36 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-u36.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=40 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-u40.c &

tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=4  -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-u4.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-u8.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=12 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-u12.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-u16.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=20 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-u20.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=24 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-u24.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=28 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-u28.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=32 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-u32.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=36 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-u36.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=40 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-u40.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=1 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u1v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=2 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u2v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=4 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u4v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=8 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u8v.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vsqrt/sse-sqrt.c.in -D BATCH_TILE=4 -o src/f32-vsqrt/gen/f32-vsqrt-sse-sqrt-u4.c &
tools/xngen src/f32-vsqrt/sse-sqrt.c.in -D BATCH_TILE=8 -o src/f32-vsqrt/gen/f32-vsqrt-sse-sqrt-u8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vsqrt/avx-sqrt.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-avx-sqrt-u8.c &
tools/xngen src/f32-vsqrt/avx-sqrt.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-avx-sqrt-u16.c &

tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-u8.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-u16.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=24 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-u24.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=32 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-u32.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=40 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-u40.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=48 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-u48.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=56 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-u56.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=64 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-u64.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=16  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-u16.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=32  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-u32.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=48  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-u48.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=64  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-u64.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=80  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-u80.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=96  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-u96.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=112 -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-u112.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=128 -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-u128.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vsqrt.yaml --output test/f32-vsqrt.cc &

wait
