#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ####################################
tools/xngen src/f32-vrsqrt/scalar-rsqrt.c.in -D BATCH_TILE=1 -o src/f32-vrsqrt/gen/f32-vrsqrt-scalar-rsqrt-u1.c &
tools/xngen src/f32-vrsqrt/scalar-rsqrt.c.in -D BATCH_TILE=2 -o src/f32-vrsqrt/gen/f32-vrsqrt-scalar-rsqrt-u2.c &
tools/xngen src/f32-vrsqrt/scalar-rsqrt.c.in -D BATCH_TILE=4 -o src/f32-vrsqrt/gen/f32-vrsqrt-scalar-rsqrt-u4.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vrsqrt/neon-rsqrt.c.in -D BATCH_TILE=4  -o src/f32-vrsqrt/gen/f32-vrsqrt-neon-rsqrt-u4.c &
tools/xngen src/f32-vrsqrt/neon-rsqrt.c.in -D BATCH_TILE=8  -o src/f32-vrsqrt/gen/f32-vrsqrt-neon-rsqrt-u8.c &
tools/xngen src/f32-vrsqrt/neon-rsqrt.c.in -D BATCH_TILE=16 -o src/f32-vrsqrt/gen/f32-vrsqrt-neon-rsqrt-u16.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vrsqrt/rvv.c.in -D LMUL=1 -o src/f32-vrsqrt/gen/f32-vrsqrt-rvv-rsqrt-u1v.c &
tools/xngen src/f32-vrsqrt/rvv.c.in -D LMUL=2 -o src/f32-vrsqrt/gen/f32-vrsqrt-rvv-rsqrt-u2v.c &
tools/xngen src/f32-vrsqrt/rvv.c.in -D LMUL=4 -o src/f32-vrsqrt/gen/f32-vrsqrt-rvv-rsqrt-u4v.c &

################################# x86 SSE ######################################
tools/xngen src/f32-vrsqrt/sse-rsqrt.c.in -D BATCH_TILE=4  -o src/f32-vrsqrt/gen/f32-vrsqrt-sse-rsqrt-u4.c &
tools/xngen src/f32-vrsqrt/sse-rsqrt.c.in -D BATCH_TILE=8  -o src/f32-vrsqrt/gen/f32-vrsqrt-sse-rsqrt-u8.c &
tools/xngen src/f32-vrsqrt/sse-rsqrt.c.in -D BATCH_TILE=16 -o src/f32-vrsqrt/gen/f32-vrsqrt-sse-rsqrt-u16.c &

################################# x86 AVX ######################################
tools/xngen src/f32-vrsqrt/avx-rsqrt.c.in -D BATCH_TILE=8  -o src/f32-vrsqrt/gen/f32-vrsqrt-avx-rsqrt-u8.c &
tools/xngen src/f32-vrsqrt/avx-rsqrt.c.in -D BATCH_TILE=16 -o src/f32-vrsqrt/gen/f32-vrsqrt-avx-rsqrt-u16.c &
tools/xngen src/f32-vrsqrt/avx-rsqrt.c.in -D BATCH_TILE=32 -o src/f32-vrsqrt/gen/f32-vrsqrt-avx-rsqrt-u32.c &

tools/xngen src/f32-vrsqrt/fma3-rsqrt.c.in -D BATCH_TILE=8  -o src/f32-vrsqrt/gen/f32-vrsqrt-fma3-rsqrt-u8.c &
tools/xngen src/f32-vrsqrt/fma3-rsqrt.c.in -D BATCH_TILE=16 -o src/f32-vrsqrt/gen/f32-vrsqrt-fma3-rsqrt-u16.c &
tools/xngen src/f32-vrsqrt/fma3-rsqrt.c.in -D BATCH_TILE=32 -o src/f32-vrsqrt/gen/f32-vrsqrt-fma3-rsqrt-u32.c &

################################# x86 AVX512 ###################################
tools/xngen src/f32-vrsqrt/avx512f-rsqrt.c.in -D BATCH_TILE=16  -o src/f32-vrsqrt/gen/f32-vrsqrt-avx512f-rsqrt-u16.c &
tools/xngen src/f32-vrsqrt/avx512f-rsqrt.c.in -D BATCH_TILE=32  -o src/f32-vrsqrt/gen/f32-vrsqrt-avx512f-rsqrt-u32.c &
tools/xngen src/f32-vrsqrt/avx512f-rsqrt.c.in -D BATCH_TILE=64  -o src/f32-vrsqrt/gen/f32-vrsqrt-avx512f-rsqrt-u64.c &

wait
