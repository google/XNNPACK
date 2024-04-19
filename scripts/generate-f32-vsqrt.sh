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
tools/xngen src/f32-vsqrt/wasmsimd-sqrt.c.in -D BATCH_TILE=4  -o src/f32-vsqrt/gen/f32-vsqrt-wasmsimd-sqrt-u4.c &
tools/xngen src/f32-vsqrt/wasmsimd-sqrt.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-wasmsimd-sqrt-u8.c &
tools/xngen src/f32-vsqrt/wasmsimd-sqrt.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-wasmsimd-sqrt-u16.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vsqrt/neon-sqrt.c.in -D BATCH_TILE=4  -o src/f32-vsqrt/gen/f32-vsqrt-aarch64-neon-sqrt-u4.c &
tools/xngen src/f32-vsqrt/neon-sqrt.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-aarch64-neon-sqrt-u8.c &
tools/xngen src/f32-vsqrt/neon-sqrt.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-aarch64-neon-sqrt-u16.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=1 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u1v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=2 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u2v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=4 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u4v.c &
tools/xngen src/f32-vsqrt/rvv-sqrt.c.in -D LMUL=8 -o src/f32-vsqrt/gen/f32-vsqrt-rvv-sqrt-u8v.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vsqrt/sse-sqrt.c.in -D BATCH_TILE=4  -o src/f32-vsqrt/gen/f32-vsqrt-sse-sqrt-u4.c &
tools/xngen src/f32-vsqrt/sse-sqrt.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-sse-sqrt-u8.c &
tools/xngen src/f32-vsqrt/sse-sqrt.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-sse-sqrt-u16.c &

tools/xngen src/f32-vsqrt/sse-rsqrt.c.in -D BATCH_TILE=4 -o src/f32-vsqrt/gen/f32-vsqrt-sse-rsqrt-u4.c &
tools/xngen src/f32-vsqrt/sse-rsqrt.c.in -D BATCH_TILE=8 -o src/f32-vsqrt/gen/f32-vsqrt-sse-rsqrt-u8.c &
tools/xngen src/f32-vsqrt/sse-rsqrt.c.in -D BATCH_TILE=12 -o src/f32-vsqrt/gen/f32-vsqrt-sse-rsqrt-u12.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vsqrt/avx-sqrt.c.in -D BATCH_TILE=8  -o src/f32-vsqrt/gen/f32-vsqrt-avx-sqrt-u8.c &
tools/xngen src/f32-vsqrt/avx-sqrt.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-avx-sqrt-u16.c &
tools/xngen src/f32-vsqrt/avx-sqrt.c.in -D BATCH_TILE=32 -o src/f32-vsqrt/gen/f32-vsqrt-avx-sqrt-u32.c &

tools/xngen src/f32-vsqrt/avx-rsqrt.c.in -D BATCH_TILE=8 -o src/f32-vsqrt/gen/f32-vsqrt-avx-rsqrt-u8.c &
tools/xngen src/f32-vsqrt/avx-rsqrt.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-avx-rsqrt-u16.c &
tools/xngen src/f32-vsqrt/avx-rsqrt.c.in -D BATCH_TILE=32 -o src/f32-vsqrt/gen/f32-vsqrt-avx-rsqrt-u32.c &

tools/xngen src/f32-vsqrt/fma3-rsqrt.c.in -D BATCH_TILE=8 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-rsqrt-u8.c &
tools/xngen src/f32-vsqrt/fma3-rsqrt.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-rsqrt-u16.c &
tools/xngen src/f32-vsqrt/fma3-rsqrt.c.in -D BATCH_TILE=32 -o src/f32-vsqrt/gen/f32-vsqrt-fma3-rsqrt-u32.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vsqrt/avx512f-rsqrt.c.in -D BATCH_TILE=16 -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-rsqrt-u16.c &
tools/xngen src/f32-vsqrt/avx512f-rsqrt.c.in -D BATCH_TILE=32 -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-rsqrt-u32.c &
tools/xngen src/f32-vsqrt/avx512f-rsqrt.c.in -D BATCH_TILE=48 -o src/f32-vsqrt/gen/f32-vsqrt-avx512f-rsqrt-u48.c &

wait
