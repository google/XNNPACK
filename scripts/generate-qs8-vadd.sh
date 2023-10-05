#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ###################################
tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-scalar-u1.c &
tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-scalar-u2.c &
tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-scalar-u4.c &

tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-scalar-u1.c &
tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-scalar-u2.c &
tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-scalar-u4.c &

tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-scalar-u1.c &
tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-scalar-u2.c &
tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-scalar-u4.c &

tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-scalar-u1.c &
tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-scalar-u2.c &
tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-scalar-u4.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-wasmsimd-u8.c &
tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-wasmsimd-u16.c &
tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-wasmsimd-u24.c &
tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-wasmsimd-u32.c &

tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-wasmsimd-u8.c &
tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-wasmsimd-u16.c &
tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-wasmsimd-u32.c &

tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-wasmsimd-u8.c &
tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-wasmsimd-u16.c &
tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-wasmsimd-u24.c &
tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-wasmsimd-u32.c &

tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-wasmsimd-u8.c &
tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-wasmsimd-u16.c &
tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-wasmsimd-u32.c &

################################### ARM NEON ##################################
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-neon-ld64-u8.c &
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-neon-ld64-u16.c &
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=24 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-neon-ld64-u24.c &
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=32 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-neon-ld64-u32.c &

tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-neon-ld128-u16.c &
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=32 -D LD128=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-neon-ld128-u32.c &

tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-neon-ld64-u8.c &
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-neon-ld64-u16.c &
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=32 -D LD128=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-neon-ld64-u32.c &

tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-neon-ld128-u16.c &

tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-neon-ld64-u8.c &
tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-neon-ld64-u16.c &
tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=24 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-neon-ld64-u24.c &
tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=32 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-neon-ld64-u32.c &

tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-neon-ld128-u16.c &
tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=32 -D LD128=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-neon-ld128-u32.c &

tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-neon-ld64-u8.c &
tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-neon-ld64-u16.c &
tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=32 -D LD128=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-neon-ld64-u32.c &

tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-neon-ld128-u16.c &

################################### x86 SSE ###################################
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse2-mul16-ld64-u8.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse2-mul16-ld64-u16.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse2-mul16-ld64-u24.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse2-mul16-ld64-u32.c &

tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-sse2-mul16-ld64-u8.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-sse2-mul16-ld64-u16.c &

tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse41-mul16-ld64-u8.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse41-mul16-ld64-u16.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse41-mul16-ld64-u24.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse41-mul16-ld64-u32.c &

tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-sse41-mul16-ld64-u8.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-sse41-mul16-ld64-u16.c &

tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul16-ld64-u8.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul16-ld64-u16.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul16-ld64-u24.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul16-ld64-u32.c &

tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-avx-mul16-ld64-u8.c &
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-avx-mul16-ld64-u16.c &

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse41-mul32-ld32-u8.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse41-mul32-ld32-u16.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse41-mul32-ld32-u24.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-sse41-mul32-ld32-u32.c &

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-sse41-mul32-ld32-u8.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-sse41-mul32-ld32-u16.c &

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul32-ld32-u8.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul32-ld32-u16.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul32-ld32-u24.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx-mul32-ld32-u32.c &

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-avx-mul32-ld32-u8.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-avx-mul32-ld32-u16.c &

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-xop-mul32-ld32-u8.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-xop-mul32-ld32-u16.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-xop-mul32-ld32-u24.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-xop-mul32-ld32-u32.c &

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-xop-mul32-ld32-u8.c &
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-xop-mul32-ld32-u16.c &

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse2-mul16-ld64-u8.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse2-mul16-ld64-u16.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse2-mul16-ld64-u24.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse2-mul16-ld64-u32.c &

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-sse2-mul16-ld64-u8.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-sse2-mul16-ld64-u16.c &

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse41-mul16-ld64-u8.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse41-mul16-ld64-u16.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse41-mul16-ld64-u24.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse41-mul16-ld64-u32.c &

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-sse41-mul16-ld64-u8.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-sse41-mul16-ld64-u16.c &

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul16-ld64-u8.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul16-ld64-u16.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul16-ld64-u24.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul16-ld64-u32.c &

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-avx-mul16-ld64-u8.c &
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-avx-mul16-ld64-u16.c &

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse41-mul32-ld32-u8.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse41-mul32-ld32-u16.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse41-mul32-ld32-u24.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-sse41-mul32-ld32-u32.c &

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-sse41-mul32-ld32-u8.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-sse41-mul32-ld32-u16.c &

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul32-ld32-u8.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul32-ld32-u16.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul32-ld32-u24.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx-mul32-ld32-u32.c &

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-avx-mul32-ld32-u8.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-avx-mul32-ld32-u16.c &

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-xop-mul32-ld32-u8.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-xop-mul32-ld32-u16.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-xop-mul32-ld32-u24.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-xop-mul32-ld32-u32.c &

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-xop-mul32-ld32-u8.c &
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-xop-mul32-ld32-u16.c &

################################### x86 AVX ###################################
tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx2-mul32-ld64-u8.c &
tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx2-mul32-ld64-u16.c &
tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx2-mul32-ld64-u24.c &
tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx2-mul32-ld64-u32.c &

tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-avx2-mul32-ld64-u8.c &
tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-avx2-mul32-ld64-u16.c &

tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx2-mul32-ld64-u8.c &
tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx2-mul32-ld64-u16.c &
tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx2-mul32-ld64-u24.c &
tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx2-mul32-ld64-u32.c &

tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-avx2-mul32-ld64-u8.c &
tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-avx2-mul32-ld64-u16.c &

################################## x86 AVX512 #################################
tools/xngen src/qs8-vadd/avx512skx-mul32-ld128.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx512skx-mul32-ld128-u16.c &
tools/xngen src/qs8-vadd/avx512skx-mul32-ld128.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vadd/gen/qs8-vadd-minmax-avx512skx-mul32-ld128-u32.c &

tools/xngen src/qs8-vadd/avx512skx-mul32-ld128.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-avx512skx-mul32-ld128-u16.c &
tools/xngen src/qs8-vadd/avx512skx-mul32-ld128.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vadd/gen/qu8-vadd-minmax-avx512skx-mul32-ld128-u32.c &

tools/xngen src/qs8-vaddc/avx512skx-mul32-ld128.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx512skx-mul32-ld128-u16.c &
tools/xngen src/qs8-vaddc/avx512skx-mul32-ld128.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/qs8-vaddc-minmax-avx512skx-mul32-ld128-u32.c &

tools/xngen src/qs8-vaddc/avx512skx-mul32-ld128.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-avx512skx-mul32-ld128-u16.c &
tools/xngen src/qs8-vaddc/avx512skx-mul32-ld128.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/qu8-vaddc-minmax-avx512skx-mul32-ld128-u32.c &

wait
