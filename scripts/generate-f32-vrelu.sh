#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vrelu/scalar.c.in -D BATCH_TILE=1 -o src/f32-vrelu/gen/f32-vrelu-scalar-u1.c &
tools/xngen src/f32-vrelu/scalar.c.in -D BATCH_TILE=2 -o src/f32-vrelu/gen/f32-vrelu-scalar-u2.c &
tools/xngen src/f32-vrelu/scalar.c.in -D BATCH_TILE=4 -o src/f32-vrelu/gen/f32-vrelu-scalar-u4.c &
tools/xngen src/f32-vrelu/scalar.c.in -D BATCH_TILE=8 -o src/f32-vrelu/gen/f32-vrelu-scalar-u8.c &

### WAsm-specific micro-kernels
tools/xngen src/f32-vrelu/wasm.c.in -D BATCH_TILE=1 -o src/f32-vrelu/gen/f32-vrelu-wasm-u1.c &
tools/xngen src/f32-vrelu/wasm.c.in -D BATCH_TILE=2 -o src/f32-vrelu/gen/f32-vrelu-wasm-u2.c &
tools/xngen src/f32-vrelu/wasm.c.in -D BATCH_TILE=4 -o src/f32-vrelu/gen/f32-vrelu-wasm-u4.c &
tools/xngen src/f32-vrelu/wasm.c.in -D BATCH_TILE=8 -o src/f32-vrelu/gen/f32-vrelu-wasm-u8.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vrelu/wasmsimd.c.in -D BATCH_TILE=4  -o src/f32-vrelu/gen/f32-vrelu-wasmsimd-u4.c &
tools/xngen src/f32-vrelu/wasmsimd.c.in -D BATCH_TILE=8  -o src/f32-vrelu/gen/f32-vrelu-wasmsimd-u8.c &
tools/xngen src/f32-vrelu/wasmsimd.c.in -D BATCH_TILE=16 -o src/f32-vrelu/gen/f32-vrelu-wasmsimd-u16.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vrelu/neon.c.in -D BATCH_TILE=4 -o src/f32-vrelu/gen/f32-vrelu-neon-u4.c &
tools/xngen src/f32-vrelu/neon.c.in -D BATCH_TILE=8 -o src/f32-vrelu/gen/f32-vrelu-neon-u8.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vrelu/sse.c.in -D BATCH_TILE=4 -o src/f32-vrelu/gen/f32-vrelu-sse-u4.c &
tools/xngen src/f32-vrelu/sse.c.in -D BATCH_TILE=8 -o src/f32-vrelu/gen/f32-vrelu-sse-u8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vrelu/avx.c.in -D BATCH_TILE=8  -o src/f32-vrelu/gen/f32-vrelu-avx-u8.c &
tools/xngen src/f32-vrelu/avx.c.in -D BATCH_TILE=16 -o src/f32-vrelu/gen/f32-vrelu-avx-u16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vrelu/avx512f.c.in -D BATCH_TILE=16 -o src/f32-vrelu/gen/f32-vrelu-avx512f-u16.c &
tools/xngen src/f32-vrelu/avx512f.c.in -D BATCH_TILE=32 -o src/f32-vrelu/gen/f32-vrelu-avx512f-u32.c &

wait
