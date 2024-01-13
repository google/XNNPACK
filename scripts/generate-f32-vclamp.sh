#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vclamp/scalar.c.in -D BATCH_TILE=1 -D WASM=0 -o src/f32-vclamp/gen/f32-vclamp-scalar-u1.c &
tools/xngen src/f32-vclamp/scalar.c.in -D BATCH_TILE=2 -D WASM=0 -o src/f32-vclamp/gen/f32-vclamp-scalar-u2.c &
tools/xngen src/f32-vclamp/scalar.c.in -D BATCH_TILE=4 -D WASM=0 -o src/f32-vclamp/gen/f32-vclamp-scalar-u4.c &

### WAsm-specific micro-kernels
tools/xngen src/f32-vclamp/scalar.c.in -D BATCH_TILE=1 -D WASM=1 -o src/f32-vclamp/gen/f32-vclamp-wasm-u1.c &
tools/xngen src/f32-vclamp/scalar.c.in -D BATCH_TILE=2 -D WASM=1 -o src/f32-vclamp/gen/f32-vclamp-wasm-u2.c &
tools/xngen src/f32-vclamp/scalar.c.in -D BATCH_TILE=4 -D WASM=1 -o src/f32-vclamp/gen/f32-vclamp-wasm-u4.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vclamp/wasmsimd.c.in -D BATCH_TILE=4 -D X86=0 -o src/f32-vclamp/gen/f32-vclamp-wasmsimd-arm-u4.c &
tools/xngen src/f32-vclamp/wasmsimd.c.in -D BATCH_TILE=8 -D X86=0 -o src/f32-vclamp/gen/f32-vclamp-wasmsimd-arm-u8.c &

tools/xngen src/f32-vclamp/wasmsimd.c.in -D BATCH_TILE=4 -D X86=1 -o src/f32-vclamp/gen/f32-vclamp-wasmsimd-x86-u4.c &
tools/xngen src/f32-vclamp/wasmsimd.c.in -D BATCH_TILE=8 -D X86=1 -o src/f32-vclamp/gen/f32-vclamp-wasmsimd-x86-u8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vclamp/neon.c.in -D BATCH_TILE=4  -o src/f32-vclamp/gen/f32-vclamp-neon-u4.c &
tools/xngen src/f32-vclamp/neon.c.in -D BATCH_TILE=8  -o src/f32-vclamp/gen/f32-vclamp-neon-u8.c &
tools/xngen src/f32-vclamp/neon.c.in -D BATCH_TILE=16 -o src/f32-vclamp/gen/f32-vclamp-neon-u16.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vclamp/rvv.c.in -D LMUL=1 -o src/f32-vclamp/gen/f32-vclamp-rvv-u1v.c &
tools/xngen src/f32-vclamp/rvv.c.in -D LMUL=2 -o src/f32-vclamp/gen/f32-vclamp-rvv-u2v.c &
tools/xngen src/f32-vclamp/rvv.c.in -D LMUL=4 -o src/f32-vclamp/gen/f32-vclamp-rvv-u4v.c &
tools/xngen src/f32-vclamp/rvv.c.in -D LMUL=8 -o src/f32-vclamp/gen/f32-vclamp-rvv-u8v.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vclamp/sse.c.in -D BATCH_TILE=4 -o src/f32-vclamp/gen/f32-vclamp-sse-u4.c &
tools/xngen src/f32-vclamp/sse.c.in -D BATCH_TILE=8 -o src/f32-vclamp/gen/f32-vclamp-sse-u8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vclamp/avx.c.in -D BATCH_TILE=8  -o src/f32-vclamp/gen/f32-vclamp-avx-u8.c &
tools/xngen src/f32-vclamp/avx.c.in -D BATCH_TILE=16 -o src/f32-vclamp/gen/f32-vclamp-avx-u16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vclamp/avx512f.c.in -D BATCH_TILE=16 -o src/f32-vclamp/gen/f32-vclamp-avx512f-u16.c &
tools/xngen src/f32-vclamp/avx512f.c.in -D BATCH_TILE=32 -o src/f32-vclamp/gen/f32-vclamp-avx512f-u32.c &

wait
