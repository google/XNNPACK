#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

############################### ARM NEONFP16ARITH ##############################
tools/xngen src/f16-vrsqrt/neonfp16arith-rsqrt.c.in -D BATCH_TILE=8 -D FULL_ACC=1 -o src/f16-vrsqrt/gen/f16-vrsqrt-neonfp16arith-rsqrt-u8.c &
tools/xngen src/f16-vrsqrt/neonfp16arith-rsqrt.c.in -D BATCH_TILE=16 -D FULL_ACC=1 -o src/f16-vrsqrt/gen/f16-vrsqrt-neonfp16arith-rsqrt-u16.c &
tools/xngen src/f16-vrsqrt/neonfp16arith-rsqrt.c.in -D BATCH_TILE=32 -D FULL_ACC=1 -o src/f16-vrsqrt/gen/f16-vrsqrt-neonfp16arith-rsqrt-u32.c &

################################### x86 F16C ##################################
tools/xngen src/f16-vrsqrt/f16c-rsqrt.c.in -D BATCH_TILE=8  -o src/f16-vrsqrt/gen/f16-vrsqrt-f16c-rsqrt-u8.c &
tools/xngen src/f16-vrsqrt/f16c-rsqrt.c.in -D BATCH_TILE=16 -o src/f16-vrsqrt/gen/f16-vrsqrt-f16c-rsqrt-u16.c &
tools/xngen src/f16-vrsqrt/f16c-rsqrt.c.in -D BATCH_TILE=32 -o src/f16-vrsqrt/gen/f16-vrsqrt-f16c-rsqrt-u32.c &

wait
