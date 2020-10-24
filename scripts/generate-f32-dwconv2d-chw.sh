#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

################################## Unit tests #################################
tools/generate-dwconv2d-chw-test.py --spec test/f32-dwconv2d-chw.yaml --output test/f32-dwconv2d-chw.cc
