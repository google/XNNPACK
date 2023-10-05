#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################

tools/xngen src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in   -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3p1-minmax-neonfp16arith-1x8.c &
tools/xngen src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in   -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3p1-minmax-neonfp16arith-2x8.c &
tools/xngen src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in   -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3p1-minmax-neonfp16arith-3x8.c &
tools/xngen src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in   -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3p1-minmax-neonfp16arith-4x8.c &
tools/xngen src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in   -D ROW_TILE=5 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3p1-minmax-neonfp16arith-5x8.c &
tools/xngen src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in   -D ROW_TILE=6 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3p1-minmax-neonfp16arith-6x8.c &

tools/xngen src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in   -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3p1-minmax-neonfp16arith-1x8-acc2.c &
tools/xngen src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in   -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3p1-minmax-neonfp16arith-1x8-acc3.c &
tools/xngen src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in   -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3p1-minmax-neonfp16arith-1x8-acc4.c &
tools/xngen src/f16-dwconv2d-chw/3x3p1-neonfp16arith.c.in   -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3p1-minmax-neonfp16arith-2x8-acc2.c &

tools/xngen src/f16-dwconv2d-chw/3x3s2p1-neonfp16arith.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3s2p1-minmax-neonfp16arith-1x8.c &
tools/xngen src/f16-dwconv2d-chw/3x3s2p1-neonfp16arith.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3s2p1-minmax-neonfp16arith-2x8.c &
tools/xngen src/f16-dwconv2d-chw/3x3s2p1-neonfp16arith.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3s2p1-minmax-neonfp16arith-3x8.c &
tools/xngen src/f16-dwconv2d-chw/3x3s2p1-neonfp16arith.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3s2p1-minmax-neonfp16arith-4x8.c &

tools/xngen src/f16-dwconv2d-chw/3x3s2p1-neonfp16arith.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3s2p1-minmax-neonfp16arith-1x8-acc2.c &
tools/xngen src/f16-dwconv2d-chw/3x3s2p1-neonfp16arith.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3s2p1-minmax-neonfp16arith-1x8-acc3.c &
tools/xngen src/f16-dwconv2d-chw/3x3s2p1-neonfp16arith.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3s2p1-minmax-neonfp16arith-1x8-acc4.c &
tools/xngen src/f16-dwconv2d-chw/3x3s2p1-neonfp16arith.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-3x3s2p1-minmax-neonfp16arith-2x8-acc2.c &

tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-1x8.c &
tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-2x8.c &
tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-3x8.c &
tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-4x8.c &
tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=5 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-5x8.c &

tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-1x8-acc2.c &
tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-1x8-acc3.c &
tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-1x8-acc4.c &
tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=1 -D ACCUMULATORS=5 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-1x8-acc5.c &
tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-2x8-acc2.c &
tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=2 -D ACCUMULATORS=3 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-2x8-acc3.c &
tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-3x8-acc2.c &
tools/xngen src/f16-dwconv2d-chw/5x5p2-neonfp16arith.c.in   -D ROW_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5p2-minmax-neonfp16arith-4x8-acc2.c &

tools/xngen src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5s2p2-minmax-neonfp16arith-1x8.c &
tools/xngen src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5s2p2-minmax-neonfp16arith-2x8.c &
tools/xngen src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5s2p2-minmax-neonfp16arith-3x8.c &

tools/xngen src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5s2p2-minmax-neonfp16arith-1x8-acc2.c &
tools/xngen src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5s2p2-minmax-neonfp16arith-1x8-acc3.c &
tools/xngen src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5s2p2-minmax-neonfp16arith-1x8-acc4.c &
tools/xngen src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5s2p2-minmax-neonfp16arith-1x8-acc5.c &
tools/xngen src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5s2p2-minmax-neonfp16arith-2x8-acc2.c &
tools/xngen src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5s2p2-minmax-neonfp16arith-2x8-acc3.c &
tools/xngen src/f16-dwconv2d-chw/5x5s2p2-neonfp16arith.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv2d-chw/gen/f16-dwconv2d-chw-5x5s2p2-minmax-neonfp16arith-3x8-acc2.c &

wait
