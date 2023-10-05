#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################# ARM NEONFP16 ################################
tools/xngen src/f16-f32acc-rsum/neonfp16.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16-u4.c &
tools/xngen src/f16-f32acc-rsum/neonfp16.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16-u8.c &
tools/xngen src/f16-f32acc-rsum/neonfp16.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16-u16-acc2.c &
tools/xngen src/f16-f32acc-rsum/neonfp16.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16-u24-acc3.c &
tools/xngen src/f16-f32acc-rsum/neonfp16.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16-u32-acc2.c &
tools/xngen src/f16-f32acc-rsum/neonfp16.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16-u32-acc4.c &

################################### x86 F16C ##################################
tools/xngen src/f16-f32acc-rsum/f16c.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-f16c-u8.c &
tools/xngen src/f16-f32acc-rsum/f16c.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-f16c-u16-acc2.c &
tools/xngen src/f16-f32acc-rsum/f16c.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-f16c-u24-acc3.c &
tools/xngen src/f16-f32acc-rsum/f16c.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-f16c-u32-acc2.c &
tools/xngen src/f16-f32acc-rsum/f16c.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-f16c-u32-acc4.c &

wait
