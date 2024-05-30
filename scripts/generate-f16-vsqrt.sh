#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEONFP16ARITH ##################################
tools/xngen src/f16-vsqrt/neonfp16arith-nr1fma1adj.c.in -D BATCH_TILE=8  -o src/f16-vsqrt/gen/f16-vsqrt-neonfp16arith-nr1fma1adj-u8.c &
tools/xngen src/f16-vsqrt/neonfp16arith-nr1fma1adj.c.in -D BATCH_TILE=16 -o src/f16-vsqrt/gen/f16-vsqrt-neonfp16arith-nr1fma1adj-u16.c &
tools/xngen src/f16-vsqrt/neonfp16arith-nr1fma1adj.c.in -D BATCH_TILE=32 -o src/f16-vsqrt/gen/f16-vsqrt-neonfp16arith-nr1fma1adj-u32.c &

################################### ARM NEON ##################################
tools/xngen src/f16-vsqrt/neonfp16arith-sqrt.c.in -D BATCH_TILE=8  -o src/f16-vsqrt/gen/f16-vsqrt-aarch64-neonfp16arith-sqrt-u8.c &
tools/xngen src/f16-vsqrt/neonfp16arith-sqrt.c.in -D BATCH_TILE=16 -o src/f16-vsqrt/gen/f16-vsqrt-aarch64-neonfp16arith-sqrt-u16.c &
tools/xngen src/f16-vsqrt/neonfp16arith-sqrt.c.in -D BATCH_TILE=32 -o src/f16-vsqrt/gen/f16-vsqrt-aarch64-neonfp16arith-sqrt-u32.c &

################################### ARM SCALAR FP16 ##################################
tools/xngen src/f16-vsqrt/fp16arith-sqrt.c.in -D BATCH_TILE=1 -o src/f16-vsqrt/gen/f16-vsqrt-fp16arith-sqrt-u1.c &
tools/xngen src/f16-vsqrt/fp16arith-sqrt.c.in -D BATCH_TILE=2 -o src/f16-vsqrt/gen/f16-vsqrt-fp16arith-sqrt-u2.c &
tools/xngen src/f16-vsqrt/fp16arith-sqrt.c.in -D BATCH_TILE=4 -o src/f16-vsqrt/gen/f16-vsqrt-fp16arith-sqrt-u4.c &

################################## x86 AVX512FP16 #################################
tools/xngen src/f16-vsqrt/avx512fp16-sqrt.c.in -D BATCH_TILE=32  -o src/f16-vsqrt/gen/f16-vsqrt-avx512fp16-sqrt-u32.c &
tools/xngen src/f16-vsqrt/avx512fp16-sqrt.c.in -D BATCH_TILE=64  -o src/f16-vsqrt/gen/f16-vsqrt-avx512fp16-sqrt-u64.c &
tools/xngen src/f16-vsqrt/avx512fp16-sqrt.c.in -D BATCH_TILE=128 -o src/f16-vsqrt/gen/f16-vsqrt-avx512fp16-sqrt-u128.c &

################################## x86 AVX512SKX #################################
tools/xngen src/f16-vsqrt/avx512skx-sqrt.c.in -D BATCH_TILE=16 -o src/f16-vsqrt/gen/f16-vsqrt-avx512skx-sqrt-u16.c &
tools/xngen src/f16-vsqrt/avx512skx-sqrt.c.in -D BATCH_TILE=32 -o src/f16-vsqrt/gen/f16-vsqrt-avx512skx-sqrt-u32.c &
tools/xngen src/f16-vsqrt/avx512skx-sqrt.c.in -D BATCH_TILE=64 -o src/f16-vsqrt/gen/f16-vsqrt-avx512skx-sqrt-u64.c &

################################### x86 F16C ##################################
tools/xngen src/f16-vsqrt/f16c-sqrt.c.in -D BATCH_TILE=8  -o src/f16-vsqrt/gen/f16-vsqrt-f16c-sqrt-u8.c &
tools/xngen src/f16-vsqrt/f16c-sqrt.c.in -D BATCH_TILE=16 -o src/f16-vsqrt/gen/f16-vsqrt-f16c-sqrt-u16.c &
tools/xngen src/f16-vsqrt/f16c-sqrt.c.in -D BATCH_TILE=32 -o src/f16-vsqrt/gen/f16-vsqrt-f16c-sqrt-u32.c &

tools/xngen src/f16-vsqrt/f16c-rsqrt.c.in -D BATCH_TILE=8  -o src/f16-vsqrt/gen/f16-vsqrt-f16c-rsqrt-u8.c &
tools/xngen src/f16-vsqrt/f16c-rsqrt.c.in -D BATCH_TILE=16 -o src/f16-vsqrt/gen/f16-vsqrt-f16c-rsqrt-u16.c &
tools/xngen src/f16-vsqrt/f16c-rsqrt.c.in -D BATCH_TILE=32 -o src/f16-vsqrt/gen/f16-vsqrt-f16c-rsqrt-u32.c &

################################## x86 AVX512FP16 #################################
tools/xngen src/f16-vsqrt/avx512fp16-sqrt.c.in -D BATCH_TILE=32  -o src/f16-vsqrt/gen/f16-vsqrt-avx512fp16-sqrt-u32.c &
tools/xngen src/f16-vsqrt/avx512fp16-sqrt.c.in -D BATCH_TILE=64  -o src/f16-vsqrt/gen/f16-vsqrt-avx512fp16-sqrt-u64.c &
tools/xngen src/f16-vsqrt/avx512fp16-sqrt.c.in -D BATCH_TILE=128 -o src/f16-vsqrt/gen/f16-vsqrt-avx512fp16-sqrt-u128.c &

wait
