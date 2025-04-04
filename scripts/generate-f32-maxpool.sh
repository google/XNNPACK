#!/bin/sh
# Copyright 2024 Imagination Technologies, inc.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f32 -D ARCH=scalar -D SIMD_SIZE=1 -o src/f32-maxpool/gen/f32-maxpool-9p-minmax-scalar-u1.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f32 -D ARCH=sse2 -D SIMD_SIZE=4 -o src/f32-maxpool/gen/f32-maxpool-9p-minmax-sse2-u4.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f32 -D ARCH=wasmsimd -D SIMD_SIZE=4 -o src/f32-maxpool/gen/f32-maxpool-9p-minmax-wasmsimd-u4.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f32 -D ARCH=neon -D SIMD_SIZE=4 -o src/f32-maxpool/gen/f32-maxpool-9p-minmax-neon-u4.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-maxpool/rvv.c.in -D LMUL=1 -o src/f32-maxpool/gen/f32-maxpool-9p-minmax-rvv-u1v.c &
tools/xngen src/f32-maxpool/rvv.c.in -D LMUL=2 -o src/f32-maxpool/gen/f32-maxpool-9p-minmax-rvv-u2v.c &

wait
