#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-1x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-2x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-3x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-4x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-5x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-6x4.c

tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neon-2x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-1x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-2x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-3x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-4x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-5x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-6x4.c

tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-neonfma-2x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-1x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-2x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-3x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-4x4.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neon-2x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-1x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-2x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-3x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-4x4.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-neonfma-2x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-1x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-2x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-3x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-4x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-5x4.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-1x4-acc5.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-2x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-2x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-3x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neon-4x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-1x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-2x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-3x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-4x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-5x4.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-1x4-acc5.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-2x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-2x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-3x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-neon.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-neonfma-4x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-1x4.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-2x4.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-3x4.c

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-1x4-acc5.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-2x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-2x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neon-3x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-1x4.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-2x4.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-3x4.c

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-1x4-acc5.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-2x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-2x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-neon.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-neonfma-3x4-acc2.c

################################### x86 SSE ###################################
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-1x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-2x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-3x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-4x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-5x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-6x4.c

tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-sse-2x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-1x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-2x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-3x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-4x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-5x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-6x4.c

tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-ssse3.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-ssse3-2x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-1x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-2x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-3x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-4x4.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-sse-2x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-1x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-2x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-3x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-4x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-5x4.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-1x4-acc5.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-2x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-2x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-3x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-sse.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-sse-4x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-1x4.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-2x4.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-3x4.c

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-1x4-acc5.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-2x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-2x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-sse.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-sse-3x4-acc2.c

################################### Scalar ####################################
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-1x1.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-2x1.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-3x1.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-4x1.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-5x1.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-6x1.c

tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-1x1-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-1x1-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-1x1-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-scalar-2x1-acc2.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-1x1.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-2x1.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-3x1.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-4x1.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-1x1-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-1x1-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-1x1-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-scalar-2x1-acc2.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-1x1.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-2x1.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-3x1.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-1x1-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-1x1-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-1x1-acc4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-1x1-acc5.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-2x1-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-2x1-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-scalar-3x1-acc2.c

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-1x1.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-2x1.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-3x1.c

tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-1x1-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-1x1-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-1x1-acc4.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-1x1-acc5.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-2x1-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-2x1-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-scalar.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv2d-chw/gen/5x5s2p2-minmax-scalar-3x1-acc2.c

################################## WAsm SIMD ##################################
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-1x4-acc3.c.in -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-wasmsimd-1x4-acc3-arm.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-1x4-acc2.c.in -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5s2p2-wasmsimd-1x4-acc2-arm.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd-1x4-acc3.c.in -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-wasmsimd-1x4-acc3-x86.c
tools/xngen src/f32-dwconv2d-chw/5x5s2p2-wasmsimd-1x4-acc2.c.in -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5s2p2-wasmsimd-1x4-acc2-x86.c

tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-arm-1x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-arm-2x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-arm-3x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-arm-4x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-arm-5x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-arm-6x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-arm-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-arm-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-arm-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-arm-2x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-x86-1x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-x86-2x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-x86-3x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-x86-4x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-x86-5x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=6 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-x86-6x4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-x86-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-x86-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-x86-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3p1-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3p1-minmax-wasmsimd-x86-2x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-arm-1x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-arm-2x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-arm-3x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-arm-4x4.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-arm-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-arm-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-arm-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-arm-2x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-x86-1x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-x86-2x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-x86-3x4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-x86-4x4.c

tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-x86-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-x86-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-x86-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/3x3s2p1-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/3x3s2p1-minmax-wasmsimd-x86-2x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-1x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-2x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-3x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-4x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-5x4.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-1x4-acc5.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-2x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-2x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-3x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -D X86=0 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-arm-4x4-acc2.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-1x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-2x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=3 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-3x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=4 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-4x4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=5 -D ACCUMULATORS=1 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-5x4.c

tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-1x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-1x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=4 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-1x4-acc4.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=1 -D ACCUMULATORS=5 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-1x4-acc5.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-2x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=2 -D ACCUMULATORS=3 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-2x4-acc3.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=3 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-3x4-acc2.c
tools/xngen src/f32-dwconv2d-chw/5x5p2-wasmsimd.c.in -D ROW_TILE=4 -D ACCUMULATORS=2 -D X86=1 -o src/f32-dwconv2d-chw/gen/5x5p2-minmax-wasmsimd-x86-4x4-acc2.c

################################## Unit tests #################################
tools/generate-dwconv2d-chw-test.py --spec test/f32-dwconv2d-chw.yaml --output test/f32-dwconv2d-chw.cc
