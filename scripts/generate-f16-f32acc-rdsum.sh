#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### NEON ######################################
tools/xngen src/f16-f32acc-rdsum/neon.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-minmax-neonfp16arith-c16.c &
tools/xngen src/f16-f32acc-rdsum/neon.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-minmax-neonfp16arith-c32.c &
tools/xngen src/f16-f32acc-rdsum/neon.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-minmax-neonfp16arith-c64.c &

wait
