#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-igemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-igemm/gen/f32-igemm-1x4-scalar.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-igemm/gen/f32-igemm-2x4-scalar.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-igemm/gen/f32-igemm-4x2-scalar.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-igemm/gen/f32-igemm-4x4-scalar.c &

tools/xngen src/f32-igemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D ACTIVATION=RELU   -o src/f32-igemm/gen/f32-igemm-1x4-relu-scalar.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D ACTIVATION=RELU   -o src/f32-igemm/gen/f32-igemm-2x4-relu-scalar.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D ACTIVATION=RELU   -o src/f32-igemm/gen/f32-igemm-4x2-relu-scalar.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D ACTIVATION=RELU   -o src/f32-igemm/gen/f32-igemm-4x4-relu-scalar.c &

tools/xngen src/f32-igemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-igemm/gen/f32-igemm-1x4-minmax-scalar.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-igemm/gen/f32-igemm-2x4-minmax-scalar.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-igemm/gen/f32-igemm-4x2-minmax-scalar.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-igemm/gen/f32-igemm-4x4-minmax-scalar.c &

### WAsm-specific micro-kernels
tools/xngen src/f32-igemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=1 -D ACTIVATION=RELU   -o src/f32-igemm/gen/f32-igemm-1x4-relu-wasm.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=1 -D ACTIVATION=RELU   -o src/f32-igemm/gen/f32-igemm-2x4-relu-wasm.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=1 -D ACTIVATION=RELU   -o src/f32-igemm/gen/f32-igemm-4x2-relu-wasm.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=1 -D ACTIVATION=RELU   -o src/f32-igemm/gen/f32-igemm-4x4-relu-wasm.c &

tools/xngen src/f32-igemm/scalar.c.in -D MR=1 -D NR=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-igemm/gen/f32-igemm-1x4-minmax-wasm.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=2 -D NR=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-igemm/gen/f32-igemm-2x4-minmax-wasm.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-igemm/gen/f32-igemm-4x2-minmax-wasm.c &
tools/xngen src/f32-igemm/scalar.c.in -D MR=4 -D NR=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-igemm/gen/f32-igemm-4x4-minmax-wasm.c &

################################## WAsm SIMD ##################################
### LOAD1+BROADCAST micro-kernels
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmsimd-arm-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmsimd-arm-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmsimd-arm-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmsimd-arm-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmsimd-arm-loadsplat.c &

tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmsimd-x86-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmsimd-x86-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmsimd-x86-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmsimd-x86-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmsimd-x86-loadsplat.c &

tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmrelaxedsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmrelaxedsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmrelaxedsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmrelaxedsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmrelaxedsimd-loadsplat.c &

tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmrelaxedsimd-fma-loadsplat.c &

tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-1x8-relu-wasmsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-3x8-relu-wasmsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-4x8-relu-wasmsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-5x8-relu-wasmsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-6x8-relu-wasmsimd-loadsplat.c &

tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8-relu-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8-relu-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8-relu-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8-relu-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8-relu-wasmrelaxedsimd-fma-loadsplat.c &

tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-1x8-wasmsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-3x8-wasmsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-4x8-wasmsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-5x8-wasmsimd-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-6x8-wasmsimd-loadsplat.c &

tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=1 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=3 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=4 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=5 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8-wasmrelaxedsimd-fma-loadsplat.c &
tools/xngen src/f32-igemm/wasmsimd-loadsplat.c.in -D MR=6 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8-wasmrelaxedsimd-fma-loadsplat.c &

### LOAD4+DUPLICATE micro-kernels
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmsimd-arm-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmsimd-arm-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmsimd-arm-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmsimd-arm-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmsimd-arm-splat.c &

tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmsimd-x86-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmsimd-x86-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmsimd-x86-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmsimd-x86-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmsimd-x86-splat.c &

tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmrelaxedsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmrelaxedsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmrelaxedsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmrelaxedsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmrelaxedsimd-splat.c &

tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8-minmax-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8-minmax-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8-minmax-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8-minmax-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8-minmax-wasmrelaxedsimd-fma-splat.c &

tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-1x8-relu-wasmsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-3x8-relu-wasmsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-4x8-relu-wasmsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-5x8-relu-wasmsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-6x8-relu-wasmsimd-splat.c &

tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8-relu-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8-relu-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8-relu-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8-relu-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8-relu-wasmrelaxedsimd-fma-splat.c &

tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-1x8-wasmsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-3x8-wasmsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-4x8-wasmsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-5x8-wasmsimd-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-6x8-wasmsimd-splat.c &

tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=1 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=3 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=4 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=5 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8-wasmrelaxedsimd-fma-splat.c &
tools/xngen src/f32-igemm/wasmsimd-splat.c.in -D MR=6 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8-wasmrelaxedsimd-fma-splat.c &

### LOAD4+PERMUTE micro-kernels
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-1x8s4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-3x8s4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-4x8s4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-5x8s4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-6x8s4-minmax-wasmsimd-arm.c &

tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-1x8s4-minmax-wasmsimd-x86.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-3x8s4-minmax-wasmsimd-x86.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-4x8s4-minmax-wasmsimd-x86.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-5x8s4-minmax-wasmsimd-x86.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-6x8s4-minmax-wasmsimd-x86.c &

tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8s4-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8s4-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8s4-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8s4-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8s4-minmax-wasmrelaxedsimd.c &

tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8s4-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8s4-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8s4-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8s4-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8s4-minmax-wasmrelaxedsimd-fma.c &

tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-1x8s4-relu-wasmsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-3x8s4-relu-wasmsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-4x8s4-relu-wasmsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-5x8s4-relu-wasmsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-6x8s4-relu-wasmsimd.c &

tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8s4-relu-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8s4-relu-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8s4-relu-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8s4-relu-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8s4-relu-wasmrelaxedsimd-fma.c &

tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-1x8s4-wasmsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-3x8s4-wasmsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-4x8s4-wasmsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-5x8s4-wasmsimd.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-6x8s4-wasmsimd.c &

tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=1 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-1x8s4-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=3 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-3x8s4-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=4 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x8s4-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=5 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-5x8s4-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/wasmsimd-s4.c.in -D MR=6 -D NR=8 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-6x8s4-wasmrelaxedsimd-fma.c &

### MRx2 micro-kernels
tools/xngen src/f32-igemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-igemm/gen/f32-igemm-4x2c4-minmax-wasmsimd-arm.c &
tools/xngen src/f32-igemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-igemm/gen/f32-igemm-4x2c4-minmax-wasmsimd-x86.c &
tools/xngen src/f32-igemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x2c4-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-igemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x2c4-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D FMA=0 -D ACTIVATION=RELU                   -o src/f32-igemm/gen/f32-igemm-4x2c4-relu-wasmsimd.c &
tools/xngen src/f32-igemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D FMA=1 -D ACTIVATION=RELU   -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x2c4-relu-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-igemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-igemm/gen/f32-igemm-4x2c4-wasmsimd.c &
tools/xngen src/f32-igemm/MRx2c4-wasmsimd.c.in -D MR=4 -D NR=2 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-igemm/gen/f32-igemm-4x2c4-wasmrelaxedsimd-fma.c &

############################### AArch64 assembly ##############################
### LD64 micro-kernels
tools/xngen src/f32-igemm/1x8-aarch64-neonfma-ld64.S.in -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-ld64.S &
tools/xngen src/f32-igemm/1x8-aarch64-neonfma-ld64.S.in -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-ld64-prfm.S &
tools/xngen src/f32-igemm/4x8-aarch64-neonfma-ld64.S.in -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-ld64.S &
tools/xngen src/f32-igemm/6x8-aarch64-neonfma-ld64.S.in -o src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-ld64.S &

### LD128 micro-kernels
tools/xngen src/f32-igemm/4x8-aarch64-neonfma-ld128.S.in -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-ld128.S &
tools/xngen src/f32-igemm/6x8-aarch64-neonfma-ld128.S.in -o src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-ld128.S &

### MRx2 micro-kernels
tools/xngen src/f32-igemm/4x2-aarch64-neonfma-ld64.S.in  -o src/f32-igemm/gen/f32-igemm-4x2-minmax-asm-aarch64-neonfma-ld64.S &
tools/xngen src/f32-igemm/4x2-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-4x2-minmax-asm-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-igemm/4x2-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-4x2-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S &

### Cortex A53 micro-kernels
tools/xngen src/f32-igemm/1x8-aarch64-neonfma-cortex-a53.S.in -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-igemm/1x8-aarch64-neonfma-cortex-a53.S.in -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S &
tools/xngen src/f32-igemm/4x8-aarch64-neonfma-cortex-a53.S.in -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-igemm/4x8-aarch64-neonfma-cortex-a53.S.in -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S &
tools/xngen src/f32-igemm/6x8-aarch64-neonfma-cortex-a53.S.in -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53.S &
tools/xngen src/f32-igemm/6x8-aarch64-neonfma-cortex-a53.S.in -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a53-prfm.S &

### Cortex A75 micro-kernels
tools/xngen src/f32-igemm/1x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-igemm/1x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S &
tools/xngen src/f32-igemm/4x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-igemm/4x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S &
tools/xngen src/f32-igemm/5x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-5x8-minmax-asm-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-igemm/5x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-5x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S &
tools/xngen src/f32-igemm/6x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a75.S &
tools/xngen src/f32-igemm/6x8-aarch64-neonfma-cortex-a75.S.in -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-asm-aarch64-neonfma-cortex-a75-prfm.S &

############################### AArch32 assembly ##############################
tools/xngen src/f32-igemm/1x8-aarch32-neon-cortex-a53.S.in       -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch32-neon-cortex-a53.S &
tools/xngen src/f32-igemm/1x8-aarch32-neon-cortex-a53.S.in       -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-asm-aarch32-neon-cortex-a53-prfm.S &
tools/xngen src/f32-igemm/4x8-aarch32-neon-cortex-a53.S.in       -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a53.S &
tools/xngen src/f32-igemm/4x8-aarch32-neon-cortex-a53.S.in       -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a53-prfm.S &
tools/xngen src/f32-igemm/4x8-aarch32-neon-cortex-a75.S.in       -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a75.S &
tools/xngen src/f32-igemm/4x8-aarch32-neon-cortex-a75.S.in       -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a75-prfm.S &
tools/xngen src/f32-igemm/4x8-aarch32-neon-cortex-a7.S.in        -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-cortex-a7.S &
tools/xngen src/f32-igemm/4x8-aarch32-neon-ld64.S.in             -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-asm-aarch32-neon-ld64.S &

################################### ARM NEON ##################################
### LD64 micro-kernels
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=1 -D NR=8 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=1 -D NR=8 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-aarch64-neonfma-lane-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=4 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-4x4-minmax-neon-lane-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=4 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-4x4-minmax-aarch64-neonfma-lane-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=8 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=8 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-aarch64-neonfma-lane-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=6 -D NR=8 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-neon-lane-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=6 -D NR=8 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-aarch64-neonfma-lane-ld64.c &

### LD128 micro-kernels
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=1 -D NR=8  -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-neon-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-neon-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=6 -D NR=8  -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-neon-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=1 -D NR=16 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-1x16-minmax-neon-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=2 -D NR=16 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-2x16-minmax-neon-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=3 -D NR=16 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-3x16-minmax-neon-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=4 -D NR=16 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-4x16-minmax-neon-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=5 -D NR=16 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-5x16-minmax-neon-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=6 -D NR=16 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-6x16-minmax-neon-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=1 -D NR=8  -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-aarch64-neonfma-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=4 -D NR=8  -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-aarch64-neonfma-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=6 -D NR=8  -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-aarch64-neonfma-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=1 -D NR=16 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-1x16-minmax-aarch64-neonfma-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=2 -D NR=16 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-2x16-minmax-aarch64-neonfma-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=3 -D NR=16 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-3x16-minmax-aarch64-neonfma-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=4 -D NR=16 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-4x16-minmax-aarch64-neonfma-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=5 -D NR=16 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-5x16-minmax-aarch64-neonfma-lane-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=6 -D NR=16 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-6x16-minmax-aarch64-neonfma-lane-ld128.c &

### MRx2 micro-kernels
tools/xngen src/f32-igemm/MRx2-neon-ld64.c.in -D MR=4 -D NR=2 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-4x2-minmax-neon-lane-ld64.c &
tools/xngen src/f32-igemm/MRx2-neon-ld64.c.in -D MR=4 -D NR=2 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-4x2-minmax-aarch64-neonfma-lane-ld64.c &
tools/xngen src/f32-igemm/MRx2-neon-ld64.c.in -D MR=6 -D NR=2 -D FMA=0 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-6x2-minmax-neon-lane-ld64.c &
tools/xngen src/f32-igemm/MRx2-neon-ld64.c.in -D MR=6 -D NR=2 -D FMA=1 -D DUP=0 -o src/f32-igemm/gen/f32-igemm-6x2-minmax-aarch64-neonfma-lane-ld64.c &
### DUP LD64 micro-kernels
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=1 -D NR=8 -D FMA=0 -D DUP=1 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=1 -D NR=8 -D FMA=1 -D DUP=1 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-neonfma-dup-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=8 -D FMA=0 -D DUP=1 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=4 -D NR=8 -D FMA=1 -D DUP=1 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-neonfma-dup-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=6 -D NR=8 -D FMA=0 -D DUP=1 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-neon-dup-ld64.c &
tools/xngen src/f32-igemm/neon-ld64.c.in      -D MR=6 -D NR=8 -D FMA=1 -D DUP=1 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-neonfma-dup-ld64.c &
### DUP LD128 micro-kernels
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=4 -D NR=8 -D FMA=0 -D DUP=1 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-neon-dup-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=4 -D NR=8 -D FMA=1 -D DUP=1 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-neonfma-dup-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=6 -D NR=8 -D FMA=0 -D DUP=1 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-neon-dup-ld128.c &
tools/xngen src/f32-igemm/neon-ld128.c.in     -D MR=6 -D NR=8 -D FMA=1 -D DUP=1 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-neonfma-dup-ld128.c &
### LOAD4+PERMUTE micro-kernels
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=1 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/f32-igemm-1x8s4-minmax-neon.c &
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=1 -D NR=8 -D FMA=1 -o src/f32-igemm/gen/f32-igemm-1x8s4-minmax-neonfma.c &
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=4 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/f32-igemm-4x8s4-minmax-neon.c &
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=4 -D NR=8 -D FMA=1 -o src/f32-igemm/gen/f32-igemm-4x8s4-minmax-neonfma.c &
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=6 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/f32-igemm-6x8s4-minmax-neon.c &
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=6 -D NR=8 -D FMA=1 -o src/f32-igemm/gen/f32-igemm-6x8s4-minmax-neonfma.c &
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=8 -D NR=8 -D FMA=0 -o src/f32-igemm/gen/f32-igemm-8x8s4-minmax-neon.c &
tools/xngen src/f32-igemm/neon-shuffle.c.in   -D MR=8 -D NR=8 -D FMA=1 -o src/f32-igemm/gen/f32-igemm-8x8s4-minmax-neonfma.c &

################################### x86 SSE ###################################
### LOAD1+BROADCAST micro-kernels
tools/xngen src/f32-igemm/sse-load1.c.in -D MR=1 -D NR=8 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-sse-load1.c &
tools/xngen src/f32-igemm/sse-load1.c.in -D MR=3 -D NR=8 -o src/f32-igemm/gen/f32-igemm-3x8-minmax-sse-load1.c &
tools/xngen src/f32-igemm/sse-load1.c.in -D MR=4 -D NR=8 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-sse-load1.c &
tools/xngen src/f32-igemm/sse-load1.c.in -D MR=5 -D NR=8 -o src/f32-igemm/gen/f32-igemm-5x8-minmax-sse-load1.c &
tools/xngen src/f32-igemm/sse-load1.c.in -D MR=6 -D NR=8 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-sse-load1.c &

### LOAD4+DUPLICATE micro-kernels
tools/xngen src/f32-igemm/sse-dup.c.in -D MR=1 -D NR=8 -D SSE=1 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-sse-dup.c &
tools/xngen src/f32-igemm/sse-dup.c.in -D MR=3 -D NR=8 -D SSE=1 -o src/f32-igemm/gen/f32-igemm-3x8-minmax-sse-dup.c &
tools/xngen src/f32-igemm/sse-dup.c.in -D MR=4 -D NR=8 -D SSE=1 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-sse-dup.c &
tools/xngen src/f32-igemm/sse-dup.c.in -D MR=5 -D NR=8 -D SSE=1 -o src/f32-igemm/gen/f32-igemm-5x8-minmax-sse-dup.c &
tools/xngen src/f32-igemm/sse-dup.c.in -D MR=6 -D NR=8 -D SSE=1 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-sse-dup.c &

### LOAD4+PERMUTE micro-kernels
tools/xngen src/f32-igemm/sse-shuffle.c.in -D MR=1 -D NR=8 -o src/f32-igemm/gen/f32-igemm-1x8s4-minmax-sse.c &
tools/xngen src/f32-igemm/sse-shuffle.c.in -D MR=3 -D NR=8 -o src/f32-igemm/gen/f32-igemm-3x8s4-minmax-sse.c &
tools/xngen src/f32-igemm/sse-shuffle.c.in -D MR=4 -D NR=8 -o src/f32-igemm/gen/f32-igemm-4x8s4-minmax-sse.c &
tools/xngen src/f32-igemm/sse-shuffle.c.in -D MR=5 -D NR=8 -o src/f32-igemm/gen/f32-igemm-5x8s4-minmax-sse.c &
tools/xngen src/f32-igemm/sse-shuffle.c.in -D MR=6 -D NR=8 -o src/f32-igemm/gen/f32-igemm-6x8s4-minmax-sse.c &

### MRx2 micro-kernels
tools/xngen src/f32-igemm/MRx2c4-sse.c.in -D MR=4 -D NR=2 -o src/f32-igemm/gen/f32-igemm-4x2c4-minmax-sse.c &
tools/xngen src/f32-igemm/MRx2c4-sse.c.in -D MR=6 -D NR=2 -o src/f32-igemm/gen/f32-igemm-6x2c4-minmax-sse.c &

################################### x86 AVX ###################################
### AVX+BROADCAST micro-kernels
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=1 -D NR=8  -D FMA=0 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-avx-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=4 -D NR=8  -D FMA=0 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-avx-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=5 -D NR=8  -D FMA=0 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-5x8-minmax-avx-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=6 -D NR=8  -D FMA=0 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-avx-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=7 -D NR=8  -D FMA=0 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-7x8-minmax-avx-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=1 -D NR=16 -D FMA=0 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-1x16-minmax-avx-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=3 -D NR=16 -D FMA=0 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-3x16-minmax-avx-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=4 -D NR=16 -D FMA=0 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-4x16-minmax-avx-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D FMA=0 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-5x16-minmax-avx-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=6 -D NR=16 -D FMA=0 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-6x16-minmax-avx-broadcast.c &
### FMA3+BROADCAST micro-kernels
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=1 -D NR=8  -D FMA=3 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-1x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=4 -D NR=8  -D FMA=3 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-4x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=5 -D NR=8  -D FMA=3 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-5x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=6 -D NR=8  -D FMA=3 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-6x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=7 -D NR=8  -D FMA=3 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-7x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=8 -D NR=8  -D FMA=3 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-8x8-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=1 -D NR=16 -D FMA=3 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-1x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=3 -D NR=16 -D FMA=3 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-3x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=4 -D NR=16 -D FMA=3 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-4x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D FMA=3 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-5x16-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=6 -D NR=16 -D FMA=3 -D PREFETCH=0 -o src/f32-igemm/gen/f32-igemm-6x16-minmax-fma3-broadcast.c &

tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=5 -D NR=16 -D FMA=3 -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-5x16-minmax-fma3-broadcast-prfm.c &
tools/xngen src/f32-igemm/avx-broadcast.c.in -D MR=6 -D NR=16 -D FMA=3 -D PREFETCH=1 -o src/f32-igemm/gen/f32-igemm-6x16-minmax-fma3-broadcast-prfm.c &

tools/xngen src/f32-igemm/avx-shuffle4.c.in -D MR=1 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/f32-igemm-1x16s4-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-shuffle4.c.in -D MR=3 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/f32-igemm-3x16s4-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-shuffle4.c.in -D MR=4 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/f32-igemm-4x16s4-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-shuffle4.c.in -D MR=5 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/f32-igemm-5x16s4-minmax-fma3-broadcast.c &
tools/xngen src/f32-igemm/avx-shuffle4.c.in -D MR=6 -D NR=16 -D FMA=3 -o src/f32-igemm/gen/f32-igemm-6x16s4-minmax-fma3-broadcast.c &

################################# x86 AVX-512 #################################
### AVX512F+BROADCAST micro-kernels
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=1 -D NR=16 -o src/f32-igemm/gen/f32-igemm-1x16-minmax-avx512f-broadcast.c &
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=4 -D NR=16 -o src/f32-igemm/gen/f32-igemm-4x16-minmax-avx512f-broadcast.c &
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=5 -D NR=16 -o src/f32-igemm/gen/f32-igemm-5x16-minmax-avx512f-broadcast.c &
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=6 -D NR=16 -o src/f32-igemm/gen/f32-igemm-6x16-minmax-avx512f-broadcast.c &
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=7 -D NR=16 -o src/f32-igemm/gen/f32-igemm-7x16-minmax-avx512f-broadcast.c &
tools/xngen src/f32-igemm/avx512-broadcast.c.in -D MR=8 -D NR=16 -o src/f32-igemm/gen/f32-igemm-8x16-minmax-avx512f-broadcast.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-igemm/MRxNRv-rvv.c.in -D MR=1 -D NR=m4 -D ACTIVATION=LINEAR -o src/f32-igemm/gen/f32-igemm-1x4v-rvv.c &
tools/xngen src/f32-igemm/MRxNRv-rvv.c.in -D MR=7 -D NR=m4 -D ACTIVATION=LINEAR -o src/f32-igemm/gen/f32-igemm-7x4v-rvv.c &
tools/xngen src/f32-igemm/MRxNRv-rvv.c.in -D MR=1 -D NR=m4 -D ACTIVATION=RELU   -o src/f32-igemm/gen/f32-igemm-1x4v-relu-rvv.c &
tools/xngen src/f32-igemm/MRxNRv-rvv.c.in -D MR=7 -D NR=m4 -D ACTIVATION=RELU   -o src/f32-igemm/gen/f32-igemm-7x4v-relu-rvv.c &
tools/xngen src/f32-igemm/MRxNRv-rvv.c.in -D MR=1 -D NR=m4 -D ACTIVATION=MINMAX -o src/f32-igemm/gen/f32-igemm-1x4v-minmax-rvv.c &
tools/xngen src/f32-igemm/MRxNRv-rvv.c.in -D MR=7 -D NR=m4 -D ACTIVATION=MINMAX -o src/f32-igemm/gen/f32-igemm-7x4v-minmax-rvv.c &

################################### HEXAGON HVX ##################################
tools/xngen src/f32-igemm/hvx-broadcast.c.in -D MR=1 -D NR=32 -o src/f32-igemm/gen/f32-igemm-1x32-minmax-hvx-broadcast.c &
tools/xngen src/f32-igemm/hvx-broadcast.c.in -D MR=8 -D NR=32 -o src/f32-igemm/gen/f32-igemm-8x32-minmax-hvx-broadcast.c &
tools/xngen src/f32-igemm/hvx-broadcast.c.in -D MR=16 -D NR=32 -o src/f32-igemm/gen/f32-igemm-16x32-minmax-hvx-broadcast.c &
tools/xngen src/f32-igemm/hvx-broadcast.c.in -D MR=1 -D NR=64 -o src/f32-igemm/gen/f32-igemm-1x64-minmax-hvx-broadcast.c &
tools/xngen src/f32-igemm/hvx-broadcast.c.in -D MR=4 -D NR=64 -o src/f32-igemm/gen/f32-igemm-4x64-minmax-hvx-broadcast.c &
tools/xngen src/f32-igemm/hvx-broadcast.c.in -D MR=7 -D NR=64 -o src/f32-igemm/gen/f32-igemm-7x64-minmax-hvx-broadcast.c &
tools/xngen src/f32-igemm/hvx-broadcast.c.in -D MR=1 -D NR=128 -o src/f32-igemm/gen/f32-igemm-1x128-minmax-hvx-broadcast.c &
tools/xngen src/f32-igemm/hvx-broadcast.c.in -D MR=2 -D NR=128 -o src/f32-igemm/gen/f32-igemm-2x128-minmax-hvx-broadcast.c &

wait
