#!/bin/sh
# Copyright 2025 Google, inc.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=u8 -D KERNEL_TILE=9 -D ARCH=scalar -D SIMD_SIZE=1 -o src/u8-maxpool/gen/u8-maxpool-9p-minmax-scalar-u1.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=u8 -D KERNEL_TILE=9 -D ARCH=sse2 -D SIMD_SIZE=16 -o src/u8-maxpool/gen/u8-maxpool-9p-minmax-sse2-u16.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=u8 -D KERNEL_TILE=9 -D ARCH=wasmsimd -D SIMD_SIZE=16 -o src/u8-maxpool/gen/u8-maxpool-9p-minmax-wasmsimd-u16.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=u8 -D KERNEL_TILE=9 -D ARCH=neon -D SIMD_SIZE=16 -o src/u8-maxpool/gen/u8-maxpool-9p-minmax-neon-u16.c &

wait
