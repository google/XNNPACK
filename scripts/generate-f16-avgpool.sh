#!/bin/sh
# Copyright 2024 Imagination Technologies, inc.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f16-avgpool/avgpool-f16-f32acc.c.in -D ARCH=neonfp16 -o src/f16-avgpool/gen/f16-f32acc-avgpool-9p-minmax-neonfp16-u4.c &
tools/xngen src/f32-avgpool/avgpool.c.in -D ARCH=neonfp16arith -D DATATYPE=f16 -D SIMD_SIZE=8 -o src/f16-avgpool/gen/f16-avgpool-9p-minmax-neonfp16arith.c &
tools/xngen src/f16-avgpool/avgpool-f16-f32acc.c.in -D ARCH=avx512f -o src/f16-avgpool/gen/f16-f32acc-avgpool-9p-minmax-avx512f-u16.c &
tools/xngen src/f32-avgpool/avgpool.c.in -D ARCH=avx512fp16 -D DATATYPE=f16 -D SIMD_SIZE=32 -o src/f16-avgpool/gen/f16-avgpool-9p-minmax-avx512fp16-u32.c &
tools/xngen src/f16-avgpool/avgpool-f16-f32acc.c.in -D ARCH=hvx -o src/f16-avgpool/gen/f16-f32acc-avgpool-9p-minmax-hvx-u32.c &

############################### Portable C/WASM ###############################
tools/xngen src/f16-avgpool/avgpool-f16-f32acc.c.in -D ARCH=scalar -o src/f16-avgpool/gen/f16-f32acc-avgpool-9p-minmax-scalar-u1.c &
tools/xngen src/f16-avgpool/avgpool-f16-f32acc.c.in -D ARCH=wasmsimd -o src/f16-avgpool/gen/f16-f32acc-avgpool-9p-minmax-wasmsimd-u4.c &
tools/xngen src/f16-avgpool/avgpool-f16-f32acc.c.in -D ARCH=wasmrelaxedsimd -o src/f16-avgpool/gen/f16-f32acc-avgpool-9p-minmax-wasmrelaxedsimd-u4.c &
tools/xngen src/f32-avgpool/avgpool.c.in -D ARCH=wasmrelaxedsimd -D DATATYPE=f16 -D SIMD_SIZE=8 -o src/f16-avgpool/gen/f16-avgpool-9p-minmax-wasmrelaxedsimdfp16-u8.c &

##################################### f16c #####################################
tools/xngen src/f16-avgpool/f16c.c.in -D SIMD_SIZE=8 -o src/f16-avgpool/gen/f16-avgpool-9p-minmax-f16c.c &

################################ RISC-V Vector #################################
tools/xngen src/f32-avgpool/rvv.c.in -D DATATYPE=f16 -D LMUL=2 -o src/f16-avgpool/gen/f16-avgpool-9p-minmax-rvvfp16arith-u2v.c &

wait
