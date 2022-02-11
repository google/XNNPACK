#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-ibilinear/neonfp16arith.c.in -D CHANNEL_TILE=8  -D PIXEL_TILE=1 -o src/f16-ibilinear/gen/neonfp16arith-c8.c &
tools/xngen src/f16-ibilinear/neonfp16arith.c.in -D CHANNEL_TILE=16 -D PIXEL_TILE=1 -o src/f16-ibilinear/gen/neonfp16arith-c16.c &

################################### x86 FMA3 ###################################
tools/xngen src/f16-ibilinear/fma3.c.in -D CHANNEL_TILE=8  -D PIXEL_TILE=1 -o src/f16-ibilinear/gen/fma3-c8.c &
tools/xngen src/f16-ibilinear/fma3.c.in -D CHANNEL_TILE=16 -D PIXEL_TILE=1 -o src/f16-ibilinear/gen/fma3-c16.c &

################################## Unit tests #################################
tools/generate-ibilinear-test.py --spec test/f16-ibilinear.yaml --output test/f16-ibilinear.cc &

wait
