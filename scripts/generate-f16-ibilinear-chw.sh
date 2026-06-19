#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

############################### ARM NEON ##############################
tools/xngen src/f16-ibilinear-chw/neonfp16arith.c.in -D PIXEL_TILE=4  -D FMA=1 -o src/f16-ibilinear-chw/gen/f16-ibilinear-chw-neonfp16arith-p4.c &
tools/xngen src/f16-ibilinear-chw/neonfp16arith.c.in -D PIXEL_TILE=8  -D FMA=1 -o src/f16-ibilinear-chw/gen/f16-ibilinear-chw-neonfp16arith-p8.c &
tools/xngen src/f16-ibilinear-chw/neonfp16arith.c.in -D PIXEL_TILE=16 -D FMA=1 -o src/f16-ibilinear-chw/gen/f16-ibilinear-chw-neonfp16arith-p16.c &

wait
