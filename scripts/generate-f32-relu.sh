#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-relu/scalar.c.in -D BATCH_TILE=1 -o src/f32-relu/gen/scalar-x1.c
tools/xngen src/f32-relu/scalar.c.in -D BATCH_TILE=2 -o src/f32-relu/gen/scalar-x2.c
tools/xngen src/f32-relu/scalar.c.in -D BATCH_TILE=4 -o src/f32-relu/gen/scalar-x4.c
tools/xngen src/f32-relu/scalar.c.in -D BATCH_TILE=8 -o src/f32-relu/gen/scalar-x8.c

### WAsm-specific micro-kernels
tools/xngen src/f32-relu/wasm.c.in -D BATCH_TILE=1 -o src/f32-relu/gen/wasm-x1.c
tools/xngen src/f32-relu/wasm.c.in -D BATCH_TILE=2 -o src/f32-relu/gen/wasm-x2.c
tools/xngen src/f32-relu/wasm.c.in -D BATCH_TILE=4 -o src/f32-relu/gen/wasm-x4.c
tools/xngen src/f32-relu/wasm.c.in -D BATCH_TILE=8 -o src/f32-relu/gen/wasm-x8.c

################################## WAsm SIMD ##################################
tools/xngen src/f32-relu/wasmsimd.c.in -D BATCH_TILE=4  -o src/f32-relu/gen/wasmsimd-x4.c
tools/xngen src/f32-relu/wasmsimd.c.in -D BATCH_TILE=8  -o src/f32-relu/gen/wasmsimd-x8.c
tools/xngen src/f32-relu/wasmsimd.c.in -D BATCH_TILE=16 -o src/f32-relu/gen/wasmsimd-x16.c

################################### ARM NEON ##################################
tools/xngen src/f32-relu/neon.c.in -D BATCH_TILE=4 -o src/f32-relu/gen/neon-x4.c
tools/xngen src/f32-relu/neon.c.in -D BATCH_TILE=8 -o src/f32-relu/gen/neon-x8.c

################################# x86 128-bit #################################
tools/xngen src/f32-relu/sse.c.in -D BATCH_TILE=4 -o src/f32-relu/gen/sse-x4.c
tools/xngen src/f32-relu/sse.c.in -D BATCH_TILE=8 -o src/f32-relu/gen/sse-x8.c

################################# x86 256-bit #################################
tools/xngen src/f32-relu/avx.c.in -D BATCH_TILE=8 -o src/f32-relu/gen/avx-x8.c
tools/xngen src/f32-relu/avx.c.in -D BATCH_TILE=16 -o src/f32-relu/gen/avx-x16.c

################################# x86 512-bit #################################
tools/xngen src/f32-relu/avx512f.c.in -D BATCH_TILE=16 -o src/f32-relu/gen/avx512f-x16.c
tools/xngen src/f32-relu/avx512f.c.in -D BATCH_TILE=32 -o src/f32-relu/gen/avx512f-x32.c

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-relu.yaml --output test/f32-relu.cc
