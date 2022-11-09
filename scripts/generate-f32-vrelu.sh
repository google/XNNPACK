#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vrelu/scalar.c.in -D BATCH_TILE=1 -o src/f32-vrelu/gen/f32-vrelu-scalar-x1.c &
tools/xngen src/f32-vrelu/scalar.c.in -D BATCH_TILE=2 -o src/f32-vrelu/gen/f32-vrelu-scalar-x2.c &
tools/xngen src/f32-vrelu/scalar.c.in -D BATCH_TILE=4 -o src/f32-vrelu/gen/f32-vrelu-scalar-x4.c &
tools/xngen src/f32-vrelu/scalar.c.in -D BATCH_TILE=8 -o src/f32-vrelu/gen/f32-vrelu-scalar-x8.c &

### WAsm-specific micro-kernels
tools/xngen src/f32-vrelu/wasm.c.in -D BATCH_TILE=1 -o src/f32-vrelu/gen/f32-vrelu-wasm-x1.c &
tools/xngen src/f32-vrelu/wasm.c.in -D BATCH_TILE=2 -o src/f32-vrelu/gen/f32-vrelu-wasm-x2.c &
tools/xngen src/f32-vrelu/wasm.c.in -D BATCH_TILE=4 -o src/f32-vrelu/gen/f32-vrelu-wasm-x4.c &
tools/xngen src/f32-vrelu/wasm.c.in -D BATCH_TILE=8 -o src/f32-vrelu/gen/f32-vrelu-wasm-x8.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vrelu/wasmsimd.c.in -D BATCH_TILE=4  -o src/f32-vrelu/gen/f32-vrelu-wasmsimd-x4.c &
tools/xngen src/f32-vrelu/wasmsimd.c.in -D BATCH_TILE=8  -o src/f32-vrelu/gen/f32-vrelu-wasmsimd-x8.c &
tools/xngen src/f32-vrelu/wasmsimd.c.in -D BATCH_TILE=16 -o src/f32-vrelu/gen/f32-vrelu-wasmsimd-x16.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vrelu/neon.c.in -D BATCH_TILE=4 -o src/f32-vrelu/gen/f32-vrelu-neon-x4.c &
tools/xngen src/f32-vrelu/neon.c.in -D BATCH_TILE=8 -o src/f32-vrelu/gen/f32-vrelu-neon-x8.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vrelu/sse.c.in -D BATCH_TILE=4 -o src/f32-vrelu/gen/f32-vrelu-sse-x4.c &
tools/xngen src/f32-vrelu/sse.c.in -D BATCH_TILE=8 -o src/f32-vrelu/gen/f32-vrelu-sse-x8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vrelu/avx.c.in -D BATCH_TILE=8  -o src/f32-vrelu/gen/f32-vrelu-avx-x8.c &
tools/xngen src/f32-vrelu/avx.c.in -D BATCH_TILE=16 -o src/f32-vrelu/gen/f32-vrelu-avx-x16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vrelu/avx512f.c.in -D BATCH_TILE=16 -o src/f32-vrelu/gen/f32-vrelu-avx512f-x16.c &
tools/xngen src/f32-vrelu/avx512f.c.in -D BATCH_TILE=32 -o src/f32-vrelu/gen/f32-vrelu-avx512f-x32.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vrelu.yaml --output test/f32-vrelu.cc &

wait
