#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=1 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-x1.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=2 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-x2.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=3 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-x3.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=4 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-x4.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=5 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-x5.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=6 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-x6.c &

tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=1 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-x1.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=2 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-x2.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=3 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-x3.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=4 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-x4.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=5 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-x5.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=6 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-x6.c &

##################################### WAsm ####################################
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=1 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-x1.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=2 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-x2.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=3 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-x3.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=4 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-x4.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=5 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-x5.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=6 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-x6.c &

tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=1 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-x1.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=2 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-x2.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=3 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-x3.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=4 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-x4.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=5 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-x5.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=6 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-x6.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-x4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-x8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-x12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-x16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-x20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-x24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-x4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-x8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-x12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-x16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-x20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-x24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-x4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-x8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-x12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-x16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-x20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-x24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-x4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-x8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-x12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-x16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-x20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-x24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=4  -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-x4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=8  -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-x8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=12 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-x12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=16 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-x16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=20 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-x20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=24 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-x24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=4  -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-x4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=8  -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-x8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=12 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-x12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=16 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-x16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=20 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-x20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=24 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-x24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=4  -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-x4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=8  -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-x8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=12 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-x12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=16 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-x16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=20 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-x20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=24 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-x24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=4  -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-x4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=8  -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-x8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=12 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-x12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=16 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-x16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=20 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-x20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=24 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-x24.c &

################################### ARM NEON ##################################
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=4  -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-x4.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=8  -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-x8.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=12 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-x12.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=16 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-x16.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=20 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-x20.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=24 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-x24.c &

tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=4  -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-x4.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=8  -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-x8.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=12 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-x12.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=16 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-x16.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=20 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-x20.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=24 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-x24.c &

tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=4  -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-x4.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=8  -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-x8.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=12 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-x12.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=16 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-x16.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=20 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-x20.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=24 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-x24.c &

tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=4  -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-x4.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=8  -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-x8.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=12 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-x12.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=16 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-x16.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=20 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-x20.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=24 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-x24.c &

################################# x86 128-bit #################################
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-x4.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-x8.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-x12.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-x16.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-x20.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-x24.c &

tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=4  -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-x4.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=8  -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-x8.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=12 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-x12.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=16 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-x16.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=20 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-x20.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=24 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-x24.c &

tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-x4.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-x8.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-x12.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-x16.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-x20.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-x24.c &

tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=4  -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-x4.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=8  -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-x8.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=12 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-x12.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=16 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-x16.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=20 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-x20.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=24 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-x24.c &

################################# x86 256-bit #################################
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-x8.c &
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-x16.c &
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-x24.c &
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-x32.c &
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-x40.c &
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-x48.c &

tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-x8.c &
tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-x16.c &
tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-x24.c &
tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-x32.c &
tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-x40.c &
tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-x48.c &

tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx-rr2-p6-x8.c &
tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx-rr2-p6-x16.c &
tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx-rr2-p6-x24.c &
tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx-rr2-p6-x32.c &
tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx-rr2-p6-x40.c &
tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx-rr2-p6-x48.c &

tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-x8.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-x16.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-x24.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-x32.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-x40.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-x48.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=56 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-x56.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=64 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-x64.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=72 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-x72.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=80 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-x80.c &

tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-x8.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-x16.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-x24.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-x32.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-x40.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-x48.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=56 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-x56.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=64 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-x64.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=72 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-x72.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=80 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-x80.c &

tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-x8.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-x16.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-x24.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-x32.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-x40.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-x48.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=56 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-x56.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=64 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-x64.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=72 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-x72.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=80 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-x80.c &

tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-x8.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-x16.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-x24.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-x32.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-x40.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-x48.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=56 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-x56.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=64 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-x64.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=72 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-x72.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=80 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-x80.c &

################################# x86 512-bit #################################
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=16  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-x16.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=32  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-x32.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=48  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-x48.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=64  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-x64.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=80  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-x80.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=96  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-x96.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=112 -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-x112.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=128 -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-x128.c &

tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=16  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-x16.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=32  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-x32.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=48  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-x48.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=64  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-x64.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=80  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-x80.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=96  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-x96.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=112 -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-x112.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=128 -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-x128.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-velu.yaml --output test/f32-velu.cc &

wait
