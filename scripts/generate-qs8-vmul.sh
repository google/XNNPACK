#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ###################################
tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-scalar-x1.c &
tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=2 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-scalar-x2.c &
tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=4 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-scalar-x4.c &

tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-scalar-x1.c &
tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=2 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-scalar-x2.c &
tools/xngen src/qs8-vmul/scalar.c.in -D BATCH_TILE=4 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-scalar-x4.c &

tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-scalar-x1.c &
tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=2 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-scalar-x2.c &
tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=4 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-scalar-x4.c &

tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-scalar-x1.c &
tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=2 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-scalar-x2.c &
tools/xngen src/qs8-vmulc/scalar.c.in -D BATCH_TILE=4 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-scalar-x4.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-vmul/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-wasmsimd-mul32-ld64-x8.c &
tools/xngen src/qs8-vmul/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-wasmsimd-mul32-ld64-x16.c &

tools/xngen src/qs8-vmul/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-wasmsimd-mul32-ld64-x8.c &
tools/xngen src/qs8-vmul/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-wasmsimd-mul32-ld64-x16.c &

tools/xngen src/qs8-vmulc/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-wasmsimd-mul32-ld64-x8.c &
tools/xngen src/qs8-vmulc/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-wasmsimd-mul32-ld64-x16.c &

tools/xngen src/qs8-vmulc/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=8  -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-wasmsimd-mul32-ld64-x8.c &
tools/xngen src/qs8-vmulc/wasmsimd-mul32-ld64.c.in -D BATCH_TILE=16 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-wasmsimd-mul32-ld64-x16.c &

################################### ARM NEON ##################################
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-neon-ld64-x8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-neon-ld64-x16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-neon-ld128-x16.c &

tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-neon-ld64-x8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-neon-ld64-x16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-neon-ld128-x16.c &

tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-neonv8-ld64-x8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-neonv8-ld64-x16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-neonv8-ld128-x16.c &

tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-neonv8-ld64-x8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-neonv8-ld64-x16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-neonv8-ld128-x16.c &

tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-rndnu-neon-ld64-x8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-rndnu-neon-ld64-x16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-rndnu-neon-ld128-x16.c &

tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-rndnu-neon-ld64-x8.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-rndnu-neon-ld64-x16.c &
tools/xngen src/qs8-vmul/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-rndnu-neon-ld128-x16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-neon-ld64-x8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-neon-ld64-x16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-neon-ld128-x16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-neon-ld64-x8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-neon-ld64-x16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-neon-ld128-x16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-neonv8-ld64-x8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-neonv8-ld64-x16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-neonv8-ld128-x16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-neonv8-ld64-x8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-neonv8-ld64-x16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=FP32 -D ARMV8=1 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-neonv8-ld128-x16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-rndnu-neon-ld64-x8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-rndnu-neon-ld64-x16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-rndnu-neon-ld128-x16.c &

tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-rndnu-neon-ld64-x8.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-rndnu-neon-ld64-x16.c &
tools/xngen src/qs8-vmulc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D REQUANTIZATION=RNDNU -D ARMV8=0 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-rndnu-neon-ld128-x16.c &

################################### x86 SSE ###################################
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-sse2-mul16-ld64-x8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-sse2-mul16-ld64-x16.c &

tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-sse2-mul16-ld64-x8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-sse2-mul16-ld64-x16.c &

tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-sse41-mul16-ld64-x8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-sse41-mul16-ld64-x16.c &

tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-sse41-mul16-ld64-x8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-sse41-mul16-ld64-x16.c &

tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-avx-mul16-ld64-x8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmul/gen/minmax-fp32-avx-mul16-ld64-x16.c &

tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-avx-mul16-ld64-x8.c &
tools/xngen src/qs8-vmul/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmul/gen/minmax-fp32-avx-mul16-ld64-x16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-sse2-mul16-ld64-x8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-sse2-mul16-ld64-x16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-sse2-mul16-ld64-x8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-sse2-mul16-ld64-x16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-sse41-mul16-ld64-x8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-sse41-mul16-ld64-x16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-sse41-mul16-ld64-x8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-sse41-mul16-ld64-x16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-avx-mul16-ld64-x8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QS8 -o src/qs8-vmulc/gen/minmax-fp32-avx-mul16-ld64-x16.c &

tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-avx-mul16-ld64-x8.c &
tools/xngen src/qs8-vmulc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D REQUANTIZATION=FP32 -D DATATYPE=QU8 -o src/qu8-vmulc/gen/minmax-fp32-avx-mul16-ld64-x16.c &

################################## Unit tests #################################
tools/generate-vbinary-test.py --tester VMulMicrokernelTester  --spec test/qs8-vmul-minmax-fp32.yaml  --output test/qs8-vmul-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VMulMicrokernelTester  --spec test/qs8-vmul-minmax-rndnu.yaml  --output test/qs8-vmul-minmax-rndnu.cc &
tools/generate-vbinary-test.py --tester VMulMicrokernelTester  --spec test/qu8-vmul-minmax-fp32.yaml  --output test/qu8-vmul-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VMulMicrokernelTester  --spec test/qu8-vmul-minmax-rndnu.yaml  --output test/qu8-vmul-minmax-rndnu.cc &

tools/generate-vbinary-test.py --tester VMulCMicrokernelTester --spec test/qs8-vmulc-minmax-fp32.yaml --output test/qs8-vmulc-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VMulCMicrokernelTester --spec test/qs8-vmulc-minmax-rndnu.yaml --output test/qs8-vmulc-minmax-rndnu.cc &
tools/generate-vbinary-test.py --tester VMulCMicrokernelTester --spec test/qu8-vmulc-minmax-fp32.yaml --output test/qu8-vmulc-minmax-fp32.cc &
tools/generate-vbinary-test.py --tester VMulCMicrokernelTester --spec test/qu8-vmulc-minmax-rndnu.yaml --output test/qu8-vmulc-minmax-rndnu.cc &

wait
