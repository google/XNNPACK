#!/bin/sh
# Copyright 2024 Imagination Technologies, inc.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f32-maxpool/maxpool.c.in -D DATATYPE=f16 -D KERNEL_TILE=9 -D ARCH=neonfp16arith -D SIMD_SIZE=8 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-neonfp16arith-u8.c &

##################################### f16c #####################################
tools/xngen src/f16-maxpool/f16c.c.in -D DATATYPE=f16 -D KERNEL_TILE=9 -D SIMD_SIZE=8 -o src/f16-maxpool/gen/f16-maxpool-9p-minmax-f16c-u8.c &

wait
