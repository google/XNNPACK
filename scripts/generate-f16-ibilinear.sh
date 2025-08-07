#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-ibilinear/neonfp16arith.c.in -D CHANNEL_TILE=8  -D PIXEL_TILE=1 -o src/f16-ibilinear/gen/f16-ibilinear-neonfp16arith-u8.c &
tools/xngen src/f16-ibilinear/neonfp16arith.c.in -D CHANNEL_TILE=16 -D PIXEL_TILE=1 -o src/f16-ibilinear/gen/f16-ibilinear-neonfp16arith-u16.c &

################################### x86 FMA3 ###################################
tools/xngen src/f16-ibilinear/fma3.c.in -D CHANNEL_TILE=8  -D PIXEL_TILE=1 -o src/f16-ibilinear/gen/f16-ibilinear-fma3-u8.c &
tools/xngen src/f16-ibilinear/fma3.c.in -D CHANNEL_TILE=16 -D PIXEL_TILE=1 -o src/f16-ibilinear/gen/f16-ibilinear-fma3-u16.c &

wait
