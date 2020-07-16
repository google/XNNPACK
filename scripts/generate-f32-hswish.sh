#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-hswish/scalar.c.in -D BATCH_TILE=1 -D WASM=0 -o src/f32-hswish/gen/hswish-scalar-x1.c
tools/xngen src/f32-hswish/scalar.c.in -D BATCH_TILE=2 -D WASM=0 -o src/f32-hswish/gen/hswish-scalar-x2.c
tools/xngen src/f32-hswish/scalar.c.in -D BATCH_TILE=4 -D WASM=0 -o src/f32-hswish/gen/hswish-scalar-x4.c

### WAsm-specific micro-kernels
tools/xngen src/f32-hswish/scalar.c.in -D BATCH_TILE=1 -D WASM=1 -o src/f32-hswish/gen/hswish-wasm-x1.c
tools/xngen src/f32-hswish/scalar.c.in -D BATCH_TILE=2 -D WASM=1 -o src/f32-hswish/gen/hswish-wasm-x2.c
tools/xngen src/f32-hswish/scalar.c.in -D BATCH_TILE=4 -D WASM=1 -o src/f32-hswish/gen/hswish-wasm-x4.c

################################## WAsm SIMD ##################################
tools/xngen src/f32-hswish/wasmsimd.c.in -D BATCH_TILE=4  -o src/f32-hswish/gen/hswish-wasmsimd-x4.c
tools/xngen src/f32-hswish/wasmsimd.c.in -D BATCH_TILE=8  -o src/f32-hswish/gen/hswish-wasmsimd-x8.c
tools/xngen src/f32-hswish/wasmsimd.c.in -D BATCH_TILE=16 -o src/f32-hswish/gen/hswish-wasmsimd-x16.c

################################### ARM NEON ##################################
tools/xngen src/f32-hswish/neon.c.in -D BATCH_TILE=4  -o src/f32-hswish/gen/hswish-neon-x4.c
tools/xngen src/f32-hswish/neon.c.in -D BATCH_TILE=8  -o src/f32-hswish/gen/hswish-neon-x8.c
tools/xngen src/f32-hswish/neon.c.in -D BATCH_TILE=16 -o src/f32-hswish/gen/hswish-neon-x16.c

################################# x86 128-bit #################################
tools/xngen src/f32-hswish/sse.c.in -D BATCH_TILE=4 -o src/f32-hswish/gen/hswish-sse-x4.c
tools/xngen src/f32-hswish/sse.c.in -D BATCH_TILE=8 -o src/f32-hswish/gen/hswish-sse-x8.c

################################# x86 256-bit #################################
tools/xngen src/f32-hswish/avx.c.in -D BATCH_TILE=8 -D FMA=0 -o src/f32-hswish/gen/hswish-avx-x8.c
tools/xngen src/f32-hswish/avx.c.in -D BATCH_TILE=16 -D FMA=0 -o src/f32-hswish/gen/hswish-avx-x16.c

tools/xngen src/f32-hswish/avx.c.in -D BATCH_TILE=8 -D FMA=3 -o src/f32-hswish/gen/hswish-fma3-x8.c
tools/xngen src/f32-hswish/avx.c.in -D BATCH_TILE=16 -D FMA=3 -o src/f32-hswish/gen/hswish-fma3-x16.c

################################# x86 512-bit #################################
tools/xngen src/f32-hswish/avx512f.c.in -D BATCH_TILE=16 -o src/f32-hswish/gen/hswish-avx512f-x16.c
tools/xngen src/f32-hswish/avx512f.c.in -D BATCH_TILE=32 -o src/f32-hswish/gen/hswish-avx512f-x32.c

################################## Unit tests #################################
tools/generate-hswish-test.py --spec test/f32-hswish.yaml --output test/f32-hswish.cc
