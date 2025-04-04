#!/bin/sh
# Copyright 2025 Google, inc.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f16 -D ARCH=neonfp16arith -D SIMD_SIZE=8 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-neonfp16arith-u8.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f16 -D ARCH=avx2 -D SIMD_SIZE=16 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-avx2-u16.c &
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f16 -D ARCH=sse41 -D SIMD_SIZE=8 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-sse41-u8.c &

wait
