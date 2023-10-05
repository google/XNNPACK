#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-neon-u8.c &
tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-neon-u16.c &
tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-neon-u32.c &

tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-neon-u8.c &
tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-neon-u16.c &
tools/xngen src/qs8-vlrelu/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-neon-u32.c &

################################### x86 SSE2 ##################################
tools/xngen src/qs8-vlrelu/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-sse2-u16.c &
tools/xngen src/qs8-vlrelu/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-sse2-u32.c &

tools/xngen src/qs8-vlrelu/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-sse2-u16.c &
tools/xngen src/qs8-vlrelu/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-sse2-u32.c &

################################## x86 SSSE3 ##################################
tools/xngen src/qs8-vlrelu/ssse3.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-ssse3-u16.c &
tools/xngen src/qs8-vlrelu/ssse3.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-ssse3-u32.c &

tools/xngen src/qs8-vlrelu/ssse3.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-ssse3-u16.c &
tools/xngen src/qs8-vlrelu/ssse3.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-ssse3-u32.c &

################################## x86 SSE4.1 #################################
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -D AVX=0 -o src/qs8-vlrelu/gen/qs8-vlrelu-sse41-u8.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -D AVX=0 -o src/qs8-vlrelu/gen/qs8-vlrelu-sse41-u16.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -D AVX=0 -o src/qs8-vlrelu/gen/qs8-vlrelu-sse41-u32.c &

tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -D AVX=0 -o src/qu8-vlrelu/gen/qu8-vlrelu-sse41-u8.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -D AVX=0 -o src/qu8-vlrelu/gen/qu8-vlrelu-sse41-u16.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -D AVX=0 -o src/qu8-vlrelu/gen/qu8-vlrelu-sse41-u32.c &

################################### x86 AVX ###################################
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -D AVX=1 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx-u8.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -D AVX=1 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx-u16.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -D AVX=1 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx-u32.c &

tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -D AVX=1 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx-u8.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -D AVX=1 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx-u16.c &
tools/xngen src/qs8-vlrelu/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -D AVX=1 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx-u32.c &

################################### x86 AVX2 ##################################
tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx2-u16.c &
tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx2-u32.c &
tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-avx2-u64.c &

tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx2-u16.c &
tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx2-u32.c &
tools/xngen src/qs8-vlrelu/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-avx2-u64.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmsimd-arm-u16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmsimd-arm-u32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmsimd-arm-u16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmsimd-arm-u32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=16 -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-arm-u16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=32 -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-arm-u32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=16 -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-arm-u16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-arm.c.in -D BATCH_TILE=32 -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-arm-u32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=8  -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmsimd-x86-u8.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmsimd-x86-u16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmsimd-x86-u32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=8  -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmsimd-x86-u8.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmsimd-x86-u16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmsimd-x86-u32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=8  -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-x86-u8.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=16 -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-x86-u16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=32 -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-wasmrelaxedsimd-x86-u32.c &

tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=8  -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-x86-u8.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=16 -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-x86-u16.c &
tools/xngen src/qs8-vlrelu/wasmsimd-x86.c.in -D BATCH_TILE=32 -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-wasmrelaxedsimd-x86-u32.c &

################################## ARMv6 SIMD #################################
tools/xngen src/qs8-vlrelu/armsimd32.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-armsimd32-u4.c &
tools/xngen src/qs8-vlrelu/armsimd32.c.in -D BATCH_TILE=8 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-armsimd32-u8.c &

tools/xngen src/qs8-vlrelu/armsimd32.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-armsimd32-u4.c &
tools/xngen src/qs8-vlrelu/armsimd32.c.in -D BATCH_TILE=8 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-armsimd32-u8.c &

#################################### Scalar ###################################
tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-select-u1.c &
tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-select-u2.c &
tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-select-u4.c &

tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-select-u1.c &
tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-select-u2.c &
tools/xngen src/qs8-vlrelu/scalar-select.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-select-u4.c &

tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-andxor-u1.c &
tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-andxor-u2.c &
tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vlrelu/gen/qs8-vlrelu-scalar-andxor-u4.c &

tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-andxor-u1.c &
tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-andxor-u2.c &
tools/xngen src/qs8-vlrelu/scalar-andxor.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vlrelu/gen/qu8-vlrelu-scalar-andxor-u4.c &

wait
