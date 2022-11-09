#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-neon-x8.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-neon-x16.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-neon-x24.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-neon-x32.c &

tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-neon-x8.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-neon-x16.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-neon-x24.c &
tools/xngen src/qs8-f32-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-neon-x32.c &

################################# x86 128-bit #################################
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse2-x8.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse2-x16.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse2-x24.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse2-x32.c &

tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse2-x8.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse2-x16.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse2-x24.c &
tools/xngen src/qs8-f32-vcvt/sse2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse2-x32.c &

tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse41-x8.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse41-x16.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse41-x24.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-sse41-x32.c &

tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse41-x8.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse41-x16.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse41-x24.c &
tools/xngen src/qs8-f32-vcvt/sse4.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-sse41-x32.c &

################################# x86 256-bit #################################
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-x8.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-x16.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-x24.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx-x32.c &

tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-x8.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-x16.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-x24.c &
tools/xngen src/qs8-f32-vcvt/avx.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx-x32.c &

tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-x8.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-x16.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-x24.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx2-x32.c &

tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-x8.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-x16.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-x24.c &
tools/xngen src/qs8-f32-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx2-x32.c &

################################# x86 512-bit #################################
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-x16.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-x32.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=48 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-x48.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=64 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-avx512skx-x64.c &

tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-x16.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-x32.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=48 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-x48.c &
tools/xngen src/qs8-f32-vcvt/avx512skx.c.in -D BATCH_TILE=64 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-avx512skx-x64.c &

################################## WAsm SIMD ##################################
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-wasmsimd-x8.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-wasmsimd-x16.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-wasmsimd-x24.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-wasmsimd-x32.c &

tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-wasmsimd-x8.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-wasmsimd-x16.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-wasmsimd-x24.c &
tools/xngen src/qs8-f32-vcvt/wasmsimd.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-wasmsimd-x32.c &

#################################### Scalar ###################################
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-scalar-x1.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-scalar-x2.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=3 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-scalar-x3.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/qs8-f32-vcvt/gen/qs8-f32-vcvt-scalar-x4.c &

tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-scalar-x1.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-scalar-x2.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=3 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-scalar-x3.c &
tools/xngen src/qs8-f32-vcvt/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/qu8-f32-vcvt/gen/qu8-f32-vcvt-scalar-x4.c &

################################## Unit tests #################################
tools/generate-vcvt-test.py --spec test/qs8-f32-vcvt.yaml --output test/qs8-f32-vcvt.cc &
tools/generate-vcvt-test.py --spec test/qu8-f32-vcvt.yaml --output test/qu8-f32-vcvt.cc &

wait
