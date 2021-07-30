#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ###################################
tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-scalar-x1.c
tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-scalar-x2.c
tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-scalar-x4.c

tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-scalar-x1.c
tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-scalar-x2.c
tools/xngen src/qs8-vadd/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-scalar-x4.c

tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-scalar-x1.c
tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-scalar-x2.c
tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-scalar-x4.c

tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-scalar-x1.c
tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-scalar-x2.c
tools/xngen src/qs8-vaddc/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-scalar-x4.c

################################## WAsm SIMD ##################################
tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-wasmsimd-x8.c
tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-wasmsimd-x16.c
tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-wasmsimd-x24.c
tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-wasmsimd-x32.c

tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-wasmsimd-x8.c
tools/xngen src/qs8-vadd/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-wasmsimd-x16.c

tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-wasmsimd-x8.c
tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-wasmsimd-x16.c
tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-wasmsimd-x24.c
tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-wasmsimd-x32.c

tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-wasmsimd-x8.c
tools/xngen src/qs8-vaddc/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-wasmsimd-x16.c

################################### ARM NEON ##################################
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-neon-ld64-x8.c
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-neon-ld64-x16.c
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=24 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-neon-ld64-x24.c
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=32 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-neon-ld64-x32.c

tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-neon-ld128-x16.c
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=32 -D LD128=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-neon-ld128-x32.c

tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-neon-ld64-x8.c
tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-neon-ld64-x16.c

tools/xngen src/qs8-vadd/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-neon-ld128-x16.c

tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-neon-ld64-x8.c
tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-neon-ld64-x16.c
tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=24 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-neon-ld64-x24.c
tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=32 -D LD128=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-neon-ld64-x32.c

tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-neon-ld128-x16.c
tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=32 -D LD128=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-neon-ld128-x32.c

tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=8  -D LD128=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-neon-ld64-x8.c
tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=16 -D LD128=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-neon-ld64-x16.c

tools/xngen src/qs8-vaddc/neon.c.in -D BATCH_TILE=16 -D LD128=1 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-neon-ld128-x16.c

################################### x86 SSE ###################################
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x8.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x16.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x24.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x32.c

tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-sse2-mul16-ld64-x8.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-sse2-mul16-ld64-x16.c

tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x8.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x16.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x24.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x32.c

tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-sse41-mul16-ld64-x8.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-sse41-mul16-ld64-x16.c

tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx-mul16-ld64-x8.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx-mul16-ld64-x16.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx-mul16-ld64-x24.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx-mul16-ld64-x32.c

tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-avx-mul16-ld64-x8.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-avx-mul16-ld64-x16.c

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse41-mul32-ld32-x8.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse41-mul32-ld32-x16.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse41-mul32-ld32-x24.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-sse41-mul32-ld32-x32.c

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-sse41-mul32-ld32-x8.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-sse41-mul32-ld32-x16.c

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx-mul32-ld32-x8.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx-mul32-ld32-x16.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx-mul32-ld32-x24.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx-mul32-ld32-x32.c

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-avx-mul32-ld32-x8.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-avx-mul32-ld32-x16.c

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-xop-mul32-ld32-x8.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-xop-mul32-ld32-x16.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-xop-mul32-ld32-x24.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-xop-mul32-ld32-x32.c

tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-xop-mul32-ld32-x8.c
tools/xngen src/qs8-vadd/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-xop-mul32-ld32-x16.c

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x8.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x16.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x24.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x32.c

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-sse2-mul16-ld64-x8.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-sse2-mul16-ld64-x16.c

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x8.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x16.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x24.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x32.c

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-sse41-mul16-ld64-x8.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-sse41-mul16-ld64-x16.c

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx-mul16-ld64-x8.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx-mul16-ld64-x16.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx-mul16-ld64-x24.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx-mul16-ld64-x32.c

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-avx-mul16-ld64-x8.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-avx-mul16-ld64-x16.c

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse41-mul32-ld32-x8.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse41-mul32-ld32-x16.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse41-mul32-ld32-x24.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-sse41-mul32-ld32-x32.c

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-sse41-mul32-ld32-x8.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=0 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-sse41-mul32-ld32-x16.c

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx-mul32-ld32-x8.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx-mul32-ld32-x16.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx-mul32-ld32-x24.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx-mul32-ld32-x32.c

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-avx-mul32-ld32-x8.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=0 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-avx-mul32-ld32-x16.c

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x8.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x16.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=24 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x24.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=32 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-xop-mul32-ld32-x32.c

tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=8  -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-xop-mul32-ld32-x8.c
tools/xngen src/qs8-vaddc/sse-mul32-ld32.c.in -D BATCH_TILE=16 -D SSE=4 -D AVX=1 -D XOP=1 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-xop-mul32-ld32-x16.c

################################### x86 AVX ###################################
tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x8.c
tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x16.c
tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x24.c
tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx2-mul32-ld64-x32.c

tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-avx2-mul32-ld64-x8.c
tools/xngen src/qs8-vadd/avx2-mul32-ld64.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-avx2-mul32-ld64-x16.c

tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x8.c
tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x16.c
tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x24.c
tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx2-mul32-ld64-x32.c

tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-avx2-mul32-ld64-x8.c
tools/xngen src/qs8-vaddc/avx2-mul32-ld64.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-avx2-mul32-ld64-x16.c

################################## x86 AVX512 #################################
tools/xngen src/qs8-vadd/avx512skx-mul32-ld128.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx512skx-mul32-ld128-x16.c
tools/xngen src/qs8-vadd/avx512skx-mul32-ld128.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vadd/gen/minmax-avx512skx-mul32-ld128-x32.c

tools/xngen src/qs8-vadd/avx512skx-mul32-ld128.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-avx512skx-mul32-ld128-x16.c
tools/xngen src/qs8-vadd/avx512skx-mul32-ld128.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vadd/gen/minmax-avx512skx-mul32-ld128-x32.c

tools/xngen src/qs8-vaddc/avx512skx-mul32-ld128.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx512skx-mul32-ld128-x16.c
tools/xngen src/qs8-vaddc/avx512skx-mul32-ld128.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-vaddc/gen/minmax-avx512skx-mul32-ld128-x32.c

tools/xngen src/qs8-vaddc/avx512skx-mul32-ld128.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-avx512skx-mul32-ld128-x16.c
tools/xngen src/qs8-vaddc/avx512skx-mul32-ld128.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-vaddc/gen/minmax-avx512skx-mul32-ld128-x32.c

################################## Unit tests #################################
tools/generate-vbinary-test.py --tester VAddMicrokernelTester  --spec test/qs8-vadd-minmax.yaml  --output test/qs8-vadd-minmax.cc
tools/generate-vbinary-test.py --tester VAddMicrokernelTester  --spec test/qu8-vadd-minmax.yaml  --output test/qu8-vadd-minmax.cc

tools/generate-vbinary-test.py --tester VAddCMicrokernelTester --spec test/qs8-vaddc-minmax.yaml --output test/qs8-vaddc-minmax.cc
tools/generate-vbinary-test.py --tester VAddCMicrokernelTester --spec test/qu8-vaddc-minmax.yaml --output test/qu8-vaddc-minmax.cc
