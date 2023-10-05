#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-neon-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-neon-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-neon-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-neon-4x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-neon-5x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-neon-6x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-neon-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-neon-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-neon-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-neon-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-aarch64-neonfma-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-aarch64-neonfma-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-aarch64-neonfma-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-aarch64-neonfma-4x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-aarch64-neonfma-5x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-aarch64-neonfma-6x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-aarch64-neonfma-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-aarch64-neonfma-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-aarch64-neonfma-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-aarch64-neonfma-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-neon-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-neon-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-neon-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-neon-4x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-neon-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-neon-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-neon-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-neon-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-aarch64-neonfma-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-aarch64-neonfma-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-aarch64-neonfma-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-aarch64-neonfma-4x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-aarch64-neonfma-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-aarch64-neonfma-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-aarch64-neonfma-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-aarch64-neonfma-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-3x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-4x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-5x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-3x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-neon-4x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-3x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-4x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-5x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-3x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-aarch64-neonfma-4x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-neon-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-neon-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-neon-3x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-neon-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-neon-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-neon-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-neon-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-neon-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-neon-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-neon-3x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-aarch64-neonfma-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-aarch64-neonfma-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-aarch64-neonfma-3x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-aarch64-neonfma-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-aarch64-neonfma-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-aarch64-neonfma-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-aarch64-neonfma-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-aarch64-neonfma-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-aarch64-neonfma-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-aarch64-neonfma-3x4-acc2.c &

################################### x86 SSE ###################################
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-sse-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-sse-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-sse-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-sse-4x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-sse-5x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-sse-6x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-sse-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-sse-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-sse-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-sse-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-4x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-5x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-6x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-ssse3-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-sse-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-sse-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-sse-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-sse-4x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-sse-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-sse-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-sse-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-sse-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-3x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-4x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-5x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-3x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-sse-4x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-sse-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-sse-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-sse-3x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-sse-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-sse-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-sse-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-sse-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-sse-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-sse-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-sse-3x4-acc2.c &

################################### Scalar ####################################
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-scalar-1x1.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-scalar-2x1.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-scalar-3x1.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-scalar-4x1.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-scalar-5x1.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-scalar-6x1.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-scalar-1x1-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-scalar-1x1-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-scalar-1x1-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-scalar-2x1-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-scalar-1x1.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-scalar-2x1.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-scalar-3x1.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-scalar-4x1.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-scalar-1x1-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-scalar-1x1-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-scalar-1x1-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-scalar-2x1-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-scalar-1x1.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-scalar-2x1.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-scalar-3x1.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-scalar-1x1-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-scalar-1x1-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-scalar-1x1-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-scalar-1x1-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-scalar-2x1-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-scalar-2x1-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-scalar-3x1-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-scalar-1x1.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-scalar-2x1.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-scalar-3x1.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-scalar-1x1-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-scalar-1x1-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-scalar-1x1-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-scalar-1x1-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-scalar-2x1-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-scalar-2x1-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-scalar-3x1-acc2.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-loadsplat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-loadsplat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-loadsplat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-loadsplat-4x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-loadsplat-5x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-loadsplat-6x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-loadsplat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-loadsplat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-loadsplat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-loadsplat-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-loadsplat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-loadsplat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-loadsplat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-loadsplat-4x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-loadsplat-5x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-loadsplat-6x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-loadsplat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-loadsplat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-loadsplat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-loadsplat-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-splat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-splat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-splat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-splat-4x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-splat-5x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-splat-6x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-splat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-splat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-splat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-arm-splat-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-splat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-splat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-splat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-splat-4x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-splat-5x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-splat-6x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-splat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-splat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-splat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3p1-minmax-wasmsimd-x86-splat-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-loadsplat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-loadsplat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-loadsplat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-loadsplat-4x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-loadsplat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-loadsplat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-loadsplat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-loadsplat-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-loadsplat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-loadsplat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-loadsplat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-loadsplat-4x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-loadsplat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-loadsplat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-loadsplat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-loadsplat-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-splat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-splat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-splat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-splat-4x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-splat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-splat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-splat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-arm-splat-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-splat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-splat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-splat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-splat-4x4.c &

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-splat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-splat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-splat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-3x3s2p1-minmax-wasmsimd-x86-splat-2x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-4x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-5x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-3x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-loadsplat-4x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-4x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-5x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-3x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-loadsplat.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-loadsplat-4x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-4x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-5x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-3x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-arm-splat-4x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-3x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-4x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-5x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-3x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd-splat.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5p2-minmax-wasmsimd-x86-splat-4x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-loadsplat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-loadsplat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-loadsplat-3x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-loadsplat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-loadsplat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-loadsplat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-loadsplat-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-loadsplat-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-loadsplat-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-loadsplat-3x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-loadsplat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-loadsplat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-loadsplat-3x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-loadsplat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-loadsplat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-loadsplat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-loadsplat-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-loadsplat-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-loadsplat-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-loadsplat.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-loadsplat-3x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-splat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-splat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-splat-3x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-splat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-splat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-splat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-splat-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-splat-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-splat-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-arm-splat-3x4-acc2.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-splat-1x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-splat-2x4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-splat-3x4.c &

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-splat-1x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-splat-1x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-splat-1x4-acc4.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-splat-1x4-acc5.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-splat-2x4-acc2.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-splat-2x4-acc3.c &
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-splat.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/f32-dwconv2d-chw-5x5s2p2-minmax-wasmsimd-x86-splat-3x4-acc2.c &

wait
