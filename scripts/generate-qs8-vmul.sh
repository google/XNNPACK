#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ###################################
tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-scalar-u1.c &
tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=2 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-scalar-u2.c &
tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=4 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-scalar-u4.c &

tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-scalar-u1.c &
tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=2 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-scalar-u2.c &
tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=4 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-scalar-u4.c &

tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-scalar-u1.c &
tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=2 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-scalar-u2.c &
tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=4 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-scalar-u4.c &

tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-scalar-u1.c &
tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=2 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-scalar-u2.c &
tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=4 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-scalar-u4.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-vmul/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-wasmsimd-mul32-ld64-u8.c &
tools/xngen src/qs8-vmul/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-wasmsimd-mul32-ld64-u16.c &

tools/xngen src/qs8-vmul/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-wasmsimd-mul32-ld64-u8.c &
tools/xngen src/qs8-vmul/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-wasmsimd-mul32-ld64-u16.c &

tools/xngen src/qs8-vmulc/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-wasmsimd-mul32-ld64-u8.c &
tools/xngen src/qs8-vmulc/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-wasmsimd-mul32-ld64-u16.c &

tools/xngen src/qs8-vmulc/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-wasmsimd-mul32-ld64-u8.c &
tools/xngen src/qs8-vmulc/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-wasmsimd-mul32-ld64-u16.c &

################################### ARM NEON ##################################
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-neon-ld64-u8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-neon-ld64-u16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-neon-ld128-u16.c &

tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-neon-ld64-u8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-neon-ld64-u16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-neon-ld128-u16.c &

tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-neonv8-ld64-u8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-neonv8-ld64-u16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-neonv8-ld128-u16.c &

tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-neonv8-ld64-u8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-neonv8-ld64-u16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-neonv8-ld128-u16.c &

tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-rndnu-neon-ld64-u8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-rndnu-neon-ld64-u16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-rndnu-neon-ld128-u16.c &

tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-rndnu-neon-ld64-u8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-rndnu-neon-ld64-u16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-rndnu-neon-ld128-u16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-neon-ld64-u8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-neon-ld64-u16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-neon-ld128-u16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-neon-ld64-u8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-neon-ld64-u16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-neon-ld128-u16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-neonv8-ld64-u8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-neonv8-ld64-u16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-neonv8-ld128-u16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-neonv8-ld64-u8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-neonv8-ld64-u16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-neonv8-ld128-u16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-rndnu-neon-ld64-u8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-rndnu-neon-ld64-u16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-rndnu-neon-ld128-u16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-rndnu-neon-ld64-u8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-rndnu-neon-ld64-u16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-rndnu-neon-ld128-u16.c &

################################### x86 SSE ###################################
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-sse2-mul16-ld64-u8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-sse2-mul16-ld64-u16.c &

tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-sse2-mul16-ld64-u8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-sse2-mul16-ld64-u16.c &

tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-sse41-mul16-ld64-u8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-sse41-mul16-ld64-u16.c &

tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-sse41-mul16-ld64-u8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-sse41-mul16-ld64-u16.c &

tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-avx-mul16-ld64-u8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-fp32-avx-mul16-ld64-u16.c &

tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-avx-mul16-ld64-u8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-fp32-avx-mul16-ld64-u16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-sse2-mul16-ld64-u8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-sse2-mul16-ld64-u16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-sse2-mul16-ld64-u8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-sse2-mul16-ld64-u16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-sse41-mul16-ld64-u8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-sse41-mul16-ld64-u16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-sse41-mul16-ld64-u8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-sse41-mul16-ld64-u16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-avx-mul16-ld64-u8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-fp32-avx-mul16-ld64-u16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-avx-mul16-ld64-u8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-fp32-avx-mul16-ld64-u16.c &

################################ RISC-V Vector ################################
tools/xngen src/qs8-vmul/rvv.c.in -D -D LMUL=1 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-f32-rvv-u1v.c &
tools/xngen src/qs8-vmul/rvv.c.in -D -D LMUL=2 -D DATATYPE=QS8 -o src/qs8-vmul/gen/qs8-vmul-minmax-f32-rvv-u2v.c &

tools/xngen src/qs8-vmul/rvv.c.in -D -D LMUL=1 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-f32-rvv-u1v.c &
tools/xngen src/qs8-vmul/rvv.c.in -D -D LMUL=2 -D DATATYPE=QU8 -o src/qu8-vmul/gen/qu8-vmul-minmax-f32-rvv-u2v.c &

tools/xngen src/qs8-vmulc/rvv.c.in -D -D LMUL=1 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-f32-rvv-u1v.c &
tools/xngen src/qs8-vmulc/rvv.c.in -D -D LMUL=2 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/qs8-vmulc-minmax-f32-rvv-u2v.c &

tools/xngen src/qs8-vmulc/rvv.c.in -D -D LMUL=1 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-f32-rvv-u1v.c &
tools/xngen src/qs8-vmulc/rvv.c.in -D -D LMUL=2 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/qu8-vmulc-minmax-f32-rvv-u2v.c &

wait
