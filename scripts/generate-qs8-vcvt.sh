#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-neon-u8.c &
tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-neon-u16.c &
tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-neon-u32.c &

tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-neon-u8.c &
tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-neon-u16.c &
tools/xngen src/qs8-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-neon-u32.c &

################################### x86 SSE2 ##################################
tools/xngen src/qs8-vcvt/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-sse2-u16.c &
tools/xngen src/qs8-vcvt/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-sse2-u32.c &

tools/xngen src/qs8-vcvt/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-sse2-u16.c &
tools/xngen src/qs8-vcvt/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-sse2-u32.c &

################################## x86 SSSE3 ##################################
tools/xngen src/qs8-vcvt/ssse3.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-ssse3-u16.c &
tools/xngen src/qs8-vcvt/ssse3.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-ssse3-u32.c &

tools/xngen src/qs8-vcvt/ssse3.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-ssse3-u16.c &
tools/xngen src/qs8-vcvt/ssse3.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-ssse3-u32.c &

################################## x86 SSE4.1 #################################
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=0 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-sse41-u8.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=0 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-sse41-u16.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=32 -D AVX=0 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-sse41-u32.c &

tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=1 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-avx-u8.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=1 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-avx-u16.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=32 -D AVX=1 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-avx-u32.c &

tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=0 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-sse41-u8.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=0 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-sse41-u16.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=32 -D AVX=0 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-sse41-u32.c &

tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=8  -D AVX=1 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-avx-u8.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=16 -D AVX=1 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-avx-u16.c &
tools/xngen src/qs8-vcvt/sse4.c.in -D BATCH_TILE=32 -D AVX=1 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-avx-u32.c &

################################### x86 AVX2 ##################################
tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-avx2-u16.c &
tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-avx2-u32.c &
tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-avx2-u64.c &

tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-avx2-u16.c &
tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-avx2-u32.c &
tools/xngen src/qs8-vcvt/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-avx2-u64.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-wasmsimd-u8.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-wasmsimd-u16.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-wasmsimd-u32.c &

tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-wasmrelaxedsimd-u8.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-wasmrelaxedsimd-u16.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D RELAXED=1 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-wasmrelaxedsimd-u32.c &

tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-wasmsimd-u8.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-wasmsimd-u16.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-wasmsimd-u32.c &

tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-wasmrelaxedsimd-u8.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-wasmrelaxedsimd-u16.c &
tools/xngen src/qs8-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D RELAXED=1 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-wasmrelaxedsimd-u32.c &

################################## ARMv6 SIMD #################################
tools/xngen src/qs8-vcvt/armsimd32.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-armsimd32-u4.c &
tools/xngen src/qs8-vcvt/armsimd32.c.in -D BATCH_TILE=8 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-armsimd32-u8.c &

tools/xngen src/qs8-vcvt/armsimd32.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-armsimd32-u4.c &
tools/xngen src/qs8-vcvt/armsimd32.c.in -D BATCH_TILE=8 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-armsimd32-u8.c &

#################################### Scalar ###################################
tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-scalar-u1.c &
tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-scalar-u2.c &
tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vcvt/gen/qs8-vcvt-scalar-u4.c &

tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-scalar-u1.c &
tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-scalar-u2.c &
tools/xngen src/qs8-vcvt/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vcvt/gen/qu8-vcvt-scalar-u4.c &

wait
