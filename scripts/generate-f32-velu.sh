#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=1 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-u1.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=2 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-u2.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=3 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-u3.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=4 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-u4.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=5 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-u5.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=6 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-lut16-p3-u6.c &

tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=1 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-u1.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=2 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-u2.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=3 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-u3.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=4 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-u4.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=5 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-u5.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=6 -D WASM=0 -o src/f32-velu/gen/f32-velu-scalar-rr2-p6-u6.c &

##################################### WAsm ####################################
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=1 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-u1.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=2 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-u2.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=3 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-u3.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=4 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-u4.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=5 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-u5.c &
tools/xngen src/f32-velu/scalar-rr2-lut16-p3.c.in -D BATCH_TILE=6 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-lut16-p3-u6.c &

tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=1 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-u1.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=2 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-u2.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=3 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-u3.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=4 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-u4.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=5 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-u5.c &
tools/xngen src/f32-velu/scalar-rr2-p6.c.in -D BATCH_TILE=6 -D WASM=1 -o src/f32-velu/gen/f32-velu-wasm-rr2-p6-u6.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-u4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-u8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-u12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-u16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-u20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-lut16-p3-u24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-u4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-u8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-u12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-u16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-u20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-lut16-p3-u24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-lut16-p3-u24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-lut16-p3-u24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=4  -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-u4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=8  -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-u8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=12 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-u12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=16 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-u16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=20 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-u20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=24 -D ARCH=ARM     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-arm-rr2-p6-u24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=4  -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-u4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=8  -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-u8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=12 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-u12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=16 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-u16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=20 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-u20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=24 -D ARCH=X86     -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmsimd-x86-rr2-p6-u24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=4  -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=8  -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=12 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=16 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=20 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=24 -D ARCH=RELAXED -D FMA=0 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-rr2-p6-u24.c &

tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=4  -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u4.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=8  -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u8.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=12 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u12.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=16 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u16.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=20 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u20.c &
tools/xngen src/f32-velu/wasmsimd-rr2-p6.c.in -D BATCH_TILE=24 -D ARCH=RELAXED -D FMA=1 -o src/f32-velu/gen/f32-velu-wasmrelaxedsimd-fma-rr2-p6-u24.c &

################################### ARM NEON ##################################
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=4  -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-u4.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=8  -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-u8.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=12 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-u12.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=16 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-u16.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=20 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-u20.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=24 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-lut16-p3-u24.c &

tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=4  -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-u4.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=8  -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-u8.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=12 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-u12.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=16 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-u16.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=20 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-u20.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=24 -D FMA=0 -o src/f32-velu/gen/f32-velu-neon-rr2-p6-u24.c &

tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=4  -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-u4.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=8  -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-u8.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=12 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-u12.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=16 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-u16.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=20 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-u20.c &
tools/xngen src/f32-velu/neon-lut16-p3.c.in -D BATCH_TILE=24 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-lut16-p3-u24.c &

tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=4  -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-u4.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=8  -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-u8.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=12 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-u12.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=16 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-u16.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=20 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-u20.c &
tools/xngen src/f32-velu/neon-p6.c.in -D BATCH_TILE=24 -D FMA=1 -o src/f32-velu/gen/f32-velu-neonfma-rr1-p6-u24.c &

################################# x86 128-bit #################################
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-u4.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-u8.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-u12.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-u16.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-u20.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-lut16-p3-u24.c &

tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=4  -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-u4.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=8  -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-u8.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=12 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-u12.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=16 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-u16.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=20 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-u20.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=24 -D SSE=2 -o src/f32-velu/gen/f32-velu-sse2-rr2-p6-u24.c &

tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=4  -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-u4.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=8  -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-u8.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=12 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-u12.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=16 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-u16.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=20 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-u20.c &
tools/xngen src/f32-velu/sse-rr2-lut16-p3.c.in -D BATCH_TILE=24 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-lut16-p3-u24.c &

tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=4  -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-u4.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=8  -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-u8.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=12 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-u12.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=16 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-u16.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=20 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-u20.c &
tools/xngen src/f32-velu/sse-rr2-p6.c.in -D BATCH_TILE=24 -D SSE=4 -o src/f32-velu/gen/f32-velu-sse41-rr2-p6-u24.c &

################################# x86 256-bit #################################
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-u8.c &
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-u16.c &
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-u24.c &
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-u32.c &
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-u40.c &
tools/xngen src/f32-velu/avx-rr2-lut4-p4-perm.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx-rr2-lut4-p4-perm-u48.c &

tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-u8.c &
tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-u16.c &
tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-u24.c &
tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-u32.c &
tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-u40.c &
tools/xngen src/f32-velu/avx-rr2-lut16-p3.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx-rr2-lut16-p3-u48.c &

tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx-rr2-p6-u8.c &
tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx-rr2-p6-u16.c &
tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx-rr2-p6-u24.c &
tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx-rr2-p6-u32.c &
tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx-rr2-p6-u40.c &
tools/xngen src/f32-velu/avx-rr2-p6.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx-rr2-p6-u48.c &

tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u8.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u16.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u24.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u32.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u40.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u48.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=56 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u56.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=64 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u64.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=72 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u72.c &
tools/xngen src/f32-velu/avx2-rr1-lut4-p4-perm.c.in -D BATCH_TILE=80 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut4-p4-perm-u80.c &

tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u8.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u16.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u24.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u32.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u40.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u48.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=56 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u56.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=64 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u64.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=72 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u72.c &
tools/xngen src/f32-velu/avx2-rr1-lut8-p4-perm.c.in -D BATCH_TILE=80 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut8-p4-perm-u80.c &

tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u8.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u16.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u24.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u32.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u40.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u48.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=56 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u56.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=64 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u64.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=72 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u72.c &
tools/xngen src/f32-velu/avx2-rr1-lut16-p3-gather.c.in -D BATCH_TILE=80 -o src/f32-velu/gen/f32-velu-avx2-rr1-lut16-p3-gather-u80.c &

tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=8  -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-u8.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=16 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-u16.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=24 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-u24.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=32 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-u32.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=40 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-u40.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=48 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-u48.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=56 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-u56.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=64 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-u64.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=72 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-u72.c &
tools/xngen src/f32-velu/avx2-rr1-p6.c.in -D BATCH_TILE=80 -o src/f32-velu/gen/f32-velu-avx2-rr1-p6-u80.c &

################################# x86 512-bit #################################
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=16  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-u16.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=32  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-u32.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=48  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-u48.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=64  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-u64.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=80  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-u80.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=96  -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-u96.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=112 -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-u112.c &
tools/xngen src/f32-velu/avx512f-rr1-lut16-p3-perm.c.in -D BATCH_TILE=128 -o src/f32-velu/gen/f32-velu-avx512f-rr1-lut16-p3-perm-u128.c &

tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=16  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-u16.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=32  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-u32.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=48  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-u48.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=64  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-u64.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=80  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-u80.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=96  -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-u96.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=112 -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-u112.c &
tools/xngen src/f32-velu/avx512f-rr1-p6.c.in -D BATCH_TILE=128 -o src/f32-velu/gen/f32-velu-avx512f-rr1-p6-u128.c &

wait
