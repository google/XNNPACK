#!/bin/sh
# Copyright 2025 Google, inc.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f16 -D ARCH=neonfp16arith -D SIMD_SIZE=8 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-neonfp16arith-u8.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f16 -D ARCH=avx512skx -D SIMD_SIZE=32 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-avx512skx-u32.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f16 -D ARCH=avx2 -D SIMD_SIZE=16 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-avx2-u16.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f16 -D ARCH=sse41 -D SIMD_SIZE=8 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-sse41-u8.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f16 -D ARCH=wasmsimd -D SIMD_SIZE=8 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-wasmsimd-u8.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f16 -D ARCH=wasmrelaxedsimd -D SIMD_SIZE=8 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-wasmrelaxedsimdfp16-u8.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f16 -D ARCH=hvx -D SIMD_SIZE=64 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-hvx-u64.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-maxpool/rvv.c.in -D DATATYPE=f16 -D LMUL=2 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-rvvfp16arith-u2v.c &

wait
