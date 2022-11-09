#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-neon-x8.c &
tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-neon-x16.c &
tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-neon-x32.c &

tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-neon-x8.c &
tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-neon-x16.c &
tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-neon-x32.c &

################################### x86 SSE2 ##################################
tools/xngen src/qs8-vlrelu/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-sse2-x16.c &
tools/xngen src/qs8-vlrelu/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-sse2-x32.c &

tools/xngen src/qs8-vlrelu/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-sse2-x16.c &
tools/xngen src/qs8-vlrelu/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-sse2-x32.c &

################################## x86 SSSE3 ##################################
tools/xngen src/qs8-vlrelu/ssse3.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-ssse3-x16.c &
tools/xngen src/qs8-vlrelu/ssse3.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-ssse3-x32.c &

tools/xngen src/qs8-vlrelu/ssse3.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-ssse3-x16.c &
tools/xngen src/qs8-vlrelu/ssse3.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-ssse3-x32.c &

################################## x86 SSE4.1 #################################
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -D AVX=0 -o src/qs8-vlrelu/gen/qs8-vlrelu-sse41-x8.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -D AVX=0 -o src/qs8-vlrelu/gen/qs8-vlrelu-sse41-x16.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -D AVX=0 -o src/qs8-vlrelu/gen/qs8-vlrelu-sse41-x32.c &

tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -D AVX=0 -o src/qu8-vlrelu/gen/qu8-vlrelu-sse41-x8.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -D AVX=0 -o src/qu8-vlrelu/gen/qu8-vlrelu-sse41-x16.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -D AVX=0 -o src/qu8-vlrelu/gen/qu8-vlrelu-sse41-x32.c &

################################### x86 AVX ###################################
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -D AVX=1 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx-x8.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -D AVX=1 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx-x16.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -D AVX=1 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx-x32.c &

tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -D AVX=1 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx-x8.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -D AVX=1 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx-x16.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -D AVX=1 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx-x32.c &

################################### x86 AVX2 ##################################
tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx2-x16.c &
tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx2-x32.c &
tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx2-x64.c &

tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx2-x16.c &
tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx2-x32.c &
tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx2-x64.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmsimd-arm-x16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmsimd-arm-x32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmsimd-arm-x16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmsimd-arm-x32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=16 -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-arm-x16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=32 -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-arm-x32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=16 -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-arm-x16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=32 -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-arm-x32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=8  -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmsimd-x86-x8.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmsimd-x86-x16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmsimd-x86-x32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=8  -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmsimd-x86-x8.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmsimd-x86-x16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmsimd-x86-x32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=8  -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-x86-x8.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=16 -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-x86-x16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=32 -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-x86-x32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=8  -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-x86-x8.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=16 -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-x86-x16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=32 -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-x86-x32.c &

################################## ARMv6 SIMD #################################
tools/xngen src/qs8-vlrelu/armsimd32.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-armsimd32-x4.c &
tools/xngen src/qs8-vlrelu/armsimd32.c.in -D BATCH_TILE=8 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-armsimd32-x8.c &

tools/xngen src/qs8-vlrelu/armsimd32.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-armsimd32-x4.c &
tools/xngen src/qs8-vlrelu/armsimd32.c.in -D BATCH_TILE=8 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-armsimd32-x8.c &

#################################### Scalar ###################################
tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-select-x1.c &
tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-select-x2.c &
tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-select-x4.c &

tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-select-x1.c &
tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-select-x2.c &
tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-select-x4.c &

tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-andxor-x1.c &
tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-andxor-x2.c &
tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-andxor-x4.c &

tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-andxor-x1.c &
tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-andxor-x2.c &
tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-andxor-x4.c &

################################## Unit tests #################################
tools/generate-vlrelu-test.py --spec test/qs8-vlrelu.yaml --output test/qs8-vlrelu.cc &
tools/generate-vlrelu-test.py --spec test/qu8-vlrelu.yaml --output test/qu8-vlrelu.cc &

wait
