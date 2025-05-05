#!/bin/sh
# Copyright 2024 Imagination Technologies, inc.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f32-avgpool/avgpool.c.in -D ARCH=scalar   -D DATATYPE=f32 -D SIMD_SIZE=1  -o src/f32-avgpool/gen/f32-avgpool-9p-minmax-scalar-u1.c &
tools/xngen src/f32-avgpool/avgpool.c.in -D ARCH=wasmsimd -D DATATYPE=f32 -D SIMD_SIZE=4  -o src/f32-avgpool/gen/f32-avgpool-9p-minmax-wasmsimd-u4.c &
tools/xngen src/f32-avgpool/avgpool.c.in -D ARCH=neon     -D DATATYPE=f32 -D SIMD_SIZE=4  -o src/f32-avgpool/gen/f32-avgpool-9p-minmax-neon-u4.c &
tools/xngen src/f32-avgpool/avgpool.c.in -D ARCH=sse2     -D DATATYPE=f32 -D SIMD_SIZE=4  -o src/f32-avgpool/gen/f32-avgpool-9p-minmax-sse2-u4.c &
tools/xngen src/f32-avgpool/avgpool.c.in -D ARCH=avx      -D DATATYPE=f32 -D SIMD_SIZE=8  -o src/f32-avgpool/gen/f32-avgpool-9p-minmax-avx-u8.c &
tools/xngen src/f32-avgpool/avgpool.c.in -D ARCH=avx512f  -D DATATYPE=f32 -D SIMD_SIZE=16 -o src/f32-avgpool/gen/f32-avgpool-9p-minmax-avx512f-u16.c &
tools/xngen src/f32-avgpool/avgpool.c.in -D ARCH=hvx      -D DATATYPE=f32 -D SIMD_SIZE=32 -o src/f32-avgpool/gen/f32-avgpool-9p-minmax-hvx-u32.c &

wait
