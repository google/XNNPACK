#!/bin/sh
# Copyright 2024 Imagination Technologies, inc.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f32-avgpool/avgpool.c.in -D ARCH=neonfp16arith -D DATATYPE=f16 -D SIMD_SIZE=8 -o src/f16-avgpool/gen/f16-avgpool-9p-minmax-neonfp16arith.c &

##################################### f16c #####################################
tools/xngen src/f16-avgpool/f16c.c.in -D SIMD_SIZE=8 -o src/f16-avgpool/gen/f16-avgpool-9p-minmax-f16c.c &

wait
