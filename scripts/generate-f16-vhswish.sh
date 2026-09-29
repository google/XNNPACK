#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f16-vhswish/simd.c.in -D ARCH=scalar -o src/f16-vhswish/gen/f16-vhswish-scalar-u1.c &

################################### ARM NEON ##################################
tools/xngen src/f16-vhswish/neonfp16arith.c.in -D BATCH_TILE=8  -o src/f16-vhswish/gen/f16-vhswish-neonfp16arith-u8.c &
tools/xngen src/f16-vhswish/neonfp16arith.c.in -D BATCH_TILE=16 -o src/f16-vhswish/gen/f16-vhswish-neonfp16arith-u16.c &

################################### x86 F16C ##################################
tools/xngen src/f16-vhswish/f16c.c.in -D BATCH_TILE=8  -o src/f16-vhswish/gen/f16-vhswish-f16c-u8.c &
tools/xngen src/f16-vhswish/f16c.c.in -D BATCH_TILE=16 -o src/f16-vhswish/gen/f16-vhswish-f16c-u16.c &

################################### RISC-V Vector #############################
tools/xngen src/f16-vhswish/rvv.c.in -D LMUL=4 -o src/f16-vhswish/gen/f16-vhswish-rvvfp16arith-u4v.c &
tools/xngen src/f16-vhswish/rvv.c.in -D LMUL=8 -o src/f16-vhswish/gen/f16-vhswish-rvvfp16arith-u8v.c &

################################ WebAssembly SIMD #############################
tools/xngen src/f16-vhswish/simd.c.in -D ARCH=wasmrelaxedsimdfp16 -o src/f16-vhswish/gen/f16-vhswish-wasmrelaxedsimdfp16-u8.c &
tools/xngen src/f16-vhswish/f16-f32acc.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILE=8 -o src/f16-vhswish/gen/f16-f32acc-vhswish-wasmrelaxedsimd-u8.c &

wait
