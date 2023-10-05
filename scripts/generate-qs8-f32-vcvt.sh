#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-neon-u8.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-neon-u16.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-neon-u24.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-neon-u32.c &

tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-neon-u8.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-neon-u16.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-neon-u24.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-neon-u32.c &

################################# x86 128-bit #################################
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse2-u8.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse2-u16.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse2-u24.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse2-u32.c &

tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse2-u8.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse2-u16.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse2-u24.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse2-u32.c &

tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse41-u8.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse41-u16.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse41-u24.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse41-u32.c &

tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse41-u8.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse41-u16.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse41-u24.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse41-u32.c &

################################# x86 256-bit #################################
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-u8.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-u16.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-u24.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-u32.c &

tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-u8.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-u16.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-u24.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-u32.c &

tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-u8.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-u16.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-u24.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-u32.c &

tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-u8.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-u16.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-u24.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-u32.c &

################################# x86 512-bit #################################
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-u16.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-u32.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=48 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-u48.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=64 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-u64.c &

tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-u16.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-u32.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=48 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-u48.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=64 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-u64.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-wasmsimd-u8.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-wasmsimd-u16.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-wasmsimd-u24.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-wasmsimd-u32.c &

tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-wasmsimd-u8.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-wasmsimd-u16.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-wasmsimd-u24.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-wasmsimd-u32.c &

#################################### Scalar ###################################
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-scalar-u1.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-scalar-u2.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=3 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-scalar-u3.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-scalar-u4.c &

tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-scalar-u1.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-scalar-u2.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=3 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-scalar-u3.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-scalar-u4.c &

wait
