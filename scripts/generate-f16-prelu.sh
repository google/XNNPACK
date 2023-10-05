#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-prelu/neonfp16arith.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -o src/f16-prelu/gen/f16-prelu-neonfp16arith-2x8.c &
tools/xngen src/f16-prelu/neonfp16arith.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f16-prelu/gen/f16-prelu-neonfp16arith-2x16.c &

################################### x86 F16C ##################################
tools/xngen src/f16-prelu/f16c.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -o src/f16-prelu/gen/f16-prelu-f16c-2x8.c &
tools/xngen src/f16-prelu/f16c.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f16-prelu/gen/f16-prelu-f16c-2x16.c &

wait
