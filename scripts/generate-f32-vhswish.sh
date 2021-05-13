#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vhswish/scalar.c.in -D BATCH_TILE=1 -D WASM=0 -o src/f32-vhswish/gen/vhswish-scalar-x1.c
tools/xngen src/f32-vhswish/scalar.c.in -D BATCH_TILE=2 -D WASM=0 -o src/f32-vhswish/gen/vhswish-scalar-x2.c
tools/xngen src/f32-vhswish/scalar.c.in -D BATCH_TILE=4 -D WASM=0 -o src/f32-vhswish/gen/vhswish-scalar-x4.c

### WAsm-specific micro-kernels
tools/xngen src/f32-vhswish/scalar.c.in -D BATCH_TILE=1 -D WASM=1 -o src/f32-vhswish/gen/vhswish-wasm-x1.c
tools/xngen src/f32-vhswish/scalar.c.in -D BATCH_TILE=2 -D WASM=1 -o src/f32-vhswish/gen/vhswish-wasm-x2.c
tools/xngen src/f32-vhswish/scalar.c.in -D BATCH_TILE=4 -D WASM=1 -o src/f32-vhswish/gen/vhswish-wasm-x4.c

################################## WAsm SIMD ##################################
tools/xngen src/f32-vhswish/wasmsimd.c.in -D BATCH_TILE=4  -o src/f32-vhswish/gen/vhswish-wasmsimd-x4.c
tools/xngen src/f32-vhswish/wasmsimd.c.in -D BATCH_TILE=8  -o src/f32-vhswish/gen/vhswish-wasmsimd-x8.c
tools/xngen src/f32-vhswish/wasmsimd.c.in -D BATCH_TILE=16 -o src/f32-vhswish/gen/vhswish-wasmsimd-x16.c

################################### ARM NEON ##################################
tools/xngen src/f32-vhswish/neon.c.in -D BATCH_TILE=4  -o src/f32-vhswish/gen/vhswish-neon-x4.c
tools/xngen src/f32-vhswish/neon.c.in -D BATCH_TILE=8  -o src/f32-vhswish/gen/vhswish-neon-x8.c
tools/xngen src/f32-vhswish/neon.c.in -D BATCH_TILE=16 -o src/f32-vhswish/gen/vhswish-neon-x16.c

################################# x86 128-bit #################################
tools/xngen src/f32-vhswish/sse.c.in -D BATCH_TILE=4 -o src/f32-vhswish/gen/vhswish-sse-x4.c
tools/xngen src/f32-vhswish/sse.c.in -D BATCH_TILE=8 -o src/f32-vhswish/gen/vhswish-sse-x8.c

################################# x86 256-bit #################################
tools/xngen src/f32-vhswish/avx.c.in -D BATCH_TILE=8 -D FMA=0 -o src/f32-vhswish/gen/vhswish-avx-x8.c
tools/xngen src/f32-vhswish/avx.c.in -D BATCH_TILE=16 -D FMA=0 -o src/f32-vhswish/gen/vhswish-avx-x16.c

tools/xngen src/f32-vhswish/avx.c.in -D BATCH_TILE=8 -D FMA=3 -o src/f32-vhswish/gen/vhswish-fma3-x8.c
tools/xngen src/f32-vhswish/avx.c.in -D BATCH_TILE=16 -D FMA=3 -o src/f32-vhswish/gen/vhswish-fma3-x16.c

################################# x86 512-bit #################################
tools/xngen src/f32-vhswish/avx512f.c.in -D BATCH_TILE=16 -o src/f32-vhswish/gen/vhswish-avx512f-x16.c
tools/xngen src/f32-vhswish/avx512f.c.in -D BATCH_TILE=32 -o src/f32-vhswish/gen/vhswish-avx512f-x32.c

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vhswish.yaml --output test/f32-vhswish.cc
