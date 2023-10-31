#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/qs8-vhswish/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-neon-u8.c &
tools/xngen src/qs8-vhswish/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-neon-u16.c &
tools/xngen src/qs8-vhswish/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-neon-u32.c &

tools/xngen src/qs8-vhswish/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-neon-u8.c &
tools/xngen src/qs8-vhswish/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-neon-u16.c &
tools/xngen src/qs8-vhswish/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-neon-u32.c &

################################### x86 SSE2 ##################################
tools/xngen src/qs8-vhswish/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-sse2-u16.c &
tools/xngen src/qs8-vhswish/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-sse2-u32.c &

tools/xngen src/qs8-vhswish/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-sse2-u16.c &
tools/xngen src/qs8-vhswish/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-sse2-u32.c &

################################## x86 SSSE3 ##################################
tools/xngen src/qs8-vhswish/ssse3.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-ssse3-u16.c &
tools/xngen src/qs8-vhswish/ssse3.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-ssse3-u32.c &

tools/xngen src/qs8-vhswish/ssse3.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-ssse3-u16.c &
tools/xngen src/qs8-vhswish/ssse3.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-ssse3-u32.c &

################################## x86 SSE4.1 #################################
tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -D AVX=0 -o src/qs8-vhswish/gen/qs8-vhswish-sse41-u8.c &
tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -D AVX=0 -o src/qs8-vhswish/gen/qs8-vhswish-sse41-u16.c &
tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -D AVX=0 -o src/qs8-vhswish/gen/qs8-vhswish-sse41-u32.c &

tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -D AVX=0 -o src/qu8-vhswish/gen/qu8-vhswish-sse41-u8.c &
tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -D AVX=0 -o src/qu8-vhswish/gen/qu8-vhswish-sse41-u16.c &
tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -D AVX=0 -o src/qu8-vhswish/gen/qu8-vhswish-sse41-u32.c &

################################## x86 AVX #################################
tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -D AVX=1 -o src/qs8-vhswish/gen/qs8-vhswish-avx-u8.c &
tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -D AVX=1 -o src/qs8-vhswish/gen/qs8-vhswish-avx-u16.c &
tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -D AVX=1 -o src/qs8-vhswish/gen/qs8-vhswish-avx-u32.c &

tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -D AVX=1 -o src/qu8-vhswish/gen/qu8-vhswish-avx-u8.c &
tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -D AVX=1 -o src/qu8-vhswish/gen/qu8-vhswish-avx-u16.c &
tools/xngen src/qs8-vhswish/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -D AVX=1 -o src/qu8-vhswish/gen/qu8-vhswish-avx-u32.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-vhswish/wasmsimd.c.in -D BATCH_TILE=8  -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-wasmsimd-u8.c &
tools/xngen src/qs8-vhswish/wasmsimd.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-wasmsimd-u16.c &
tools/xngen src/qs8-vhswish/wasmsimd.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-wasmsimd-u32.c &

tools/xngen src/qs8-vhswish/wasmsimd.c.in -D BATCH_TILE=8  -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-wasmsimd-u8.c &
tools/xngen src/qs8-vhswish/wasmsimd.c.in -D BATCH_TILE=16 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-wasmsimd-u16.c &
tools/xngen src/qs8-vhswish/wasmsimd.c.in -D BATCH_TILE=32 -D RELAXED=0 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-wasmsimd-u32.c &

#################################### Scalar ###################################
tools/xngen src/qs8-vhswish/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-scalar-u1.c &
tools/xngen src/qs8-vhswish/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-scalar-u2.c &
tools/xngen src/qs8-vhswish/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vhswish/gen/qs8-vhswish-scalar-u4.c &

tools/xngen src/qs8-vhswish/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-scalar-u1.c &
tools/xngen src/qs8-vhswish/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-scalar-u2.c &
tools/xngen src/qs8-vhswish/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vhswish/gen/qu8-vhswish-scalar-u4.c &

wait
