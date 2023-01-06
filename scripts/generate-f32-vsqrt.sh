#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-vsqrt/scalar-sqrt.c.in -D BATCH_TILE=1 -o src/f32-vsqrt/gen/f32-vsqrt-scalar-sqrt-x1.c &
tools/xngen src/f32-vsqrt/scalar-sqrt.c.in -D BATCH_TILE=2 -o src/f32-vsqrt/gen/f32-vsqrt-scalar-sqrt-x2.c &
tools/xngen src/f32-vsqrt/scalar-sqrt.c.in -D BATCH_TILE=4 -o src/f32-vsqrt/gen/f32-vsqrt-scalar-sqrt-x4.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vsqrt/wasmsimd-sqrt.c.in -D BATCH_TILE=4 -o src/f32-vsqrt/gen/f32-vsqrt-wasmsimd-sqrt-x4.c &
tools/xngen src/f32-vsqrt/wasmsimd-sqrt.c.in -D BATCH_TILE=8 -o src/f32-vsqrt/gen/f32-vsqrt-wasmsimd-sqrt-x8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vsqrt/neon-sqrt.c.in -D BATCH_TILE=4 -o src/f32-vsqrt/gen/f32-vsqrt-aarch64-neon-sqrt-x4.c &
tools/xngen src/f32-vsqrt/neon-sqrt.c.in -D BATCH_TILE=8 -o src/f32-vsqrt/gen/f32-vsqrt-aarch64-neon-sqrt-x8.c &

tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=4  -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-x4.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-x8.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=12 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-x12.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-x16.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=20 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-x20.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=24 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-x24.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=28 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-x28.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=32 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-x32.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=36 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-x36.c &
tools/xngen src/f32-vsqrt/neonfma-nr2fma1adj.c.in -D BATCH_TILE=40 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr2fma1adj-x40.c &

tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=4  -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-x4.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-x8.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=12 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-x12.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-x16.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=20 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-x20.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=24 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-x24.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=28 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-x28.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=32 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-x32.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=36 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-x36.c &
tools/xngen src/f32-vsqrt/neonfma-nr1rsqrts1fma1adj.c.in -D BATCH_TILE=40 -o src/f32-vsqrt/gen/f32-vsqrt-neonfma-nr1rsqrts1fma1adj-x40.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=1 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-x1v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=2 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-x2v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=4 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-x4v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=8 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-x8v.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vsqrt/sse-sqrt.c.in -D BATCH_TILE=4 -o src/f32-vsqrt/gen/f32-vsqrt-sse-sqrt-x4.c &
tools/xngen src/f32-vsqrt/sse-sqrt.c.in -D BATCH_TILE=8 -o src/f32-vsqrt/gen/f32-vsqrt-sse-sqrt-x8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vsqrt/avx-sqrt.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-avx-sqrt-x8.c &
tools/xngen src/f32-vsqrt/avx-sqrt.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-avx-sqrt-x16.c &

tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-x8.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-x16.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=24 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-x24.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=32 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-x32.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=40 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-x40.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=48 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-x48.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=56 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-x56.c &
tools/xngen src/f32-vsqrt/fma3-nr1fma1adj.c.in -D BATCH_TILE=64 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-nr1fma1adj-x64.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=16  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-x16.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=32  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-x32.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=48  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-x48.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=64  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-x64.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=80  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-x80.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=96  -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-x96.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=112 -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-x112.c &
tools/xngen src/f32-vsqrt/avx512f-nr1fma1adj.c.in -D BATCH_TILE=128 -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-nr1fma1adj-x128.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vsqrt.yaml --output test/f32-vsqrt.cc &

wait
