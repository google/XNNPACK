#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-sigmoid/neonfma-p5-nr2fma.c.in -D BATCH_TILE=16 -o src/f32-sigmoid/gen/neonfma-p5-nr2fma-x16.c
tools/xngen src/f32-sigmoid/neon-frac-p9-p10-nr1recps.c.in -D BATCH_TILE=16 -o src/f32-sigmoid/gen/neon-frac-p9-p10-nr1recps-x16.c

################################### x86 SSE ###################################
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=8 -D BLEND=0 -o src/f32-sigmoid/gen/sse2-p5-div-x8.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=16 -D BLEND=0 -o src/f32-sigmoid/gen/sse2-p5-div-x16.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=8 -D BLEND=1 -o src/f32-sigmoid/gen/sse41-p5-div-x8.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=16 -D BLEND=1 -o src/f32-sigmoid/gen/sse41-p5-div-x16.c

################################### Scalar ####################################
tools/xngen src/f32-sigmoid/scalar-lut2048-p1-div.c.in -D BATCH_TILE=1 -o src/f32-sigmoid/gen/scalar-lut2048-p1-div-x1.c
tools/xngen src/f32-sigmoid/scalar-lut2048-p1-div.c.in -D BATCH_TILE=2 -o src/f32-sigmoid/gen/scalar-lut2048-p1-div-x2.c
tools/xngen src/f32-sigmoid/scalar-lut2048-p1-div.c.in -D BATCH_TILE=4 -o src/f32-sigmoid/gen/scalar-lut2048-p1-div-x4.c

tools/xngen src/f32-sigmoid/scalar-lut64-p2-div.c.in -D BATCH_TILE=1 -o src/f32-sigmoid/gen/scalar-lut64-p2-div-x1.c
tools/xngen src/f32-sigmoid/scalar-lut64-p2-div.c.in -D BATCH_TILE=2 -o src/f32-sigmoid/gen/scalar-lut64-p2-div-x2.c
tools/xngen src/f32-sigmoid/scalar-lut64-p2-div.c.in -D BATCH_TILE=4 -o src/f32-sigmoid/gen/scalar-lut64-p2-div-x4.c

tools/xngen src/f32-sigmoid/scalar-p5-div.c.in -D BATCH_TILE=1 -o src/f32-sigmoid/gen/scalar-p5-div-x1.c
tools/xngen src/f32-sigmoid/scalar-p5-div.c.in -D BATCH_TILE=2 -o src/f32-sigmoid/gen/scalar-p5-div-x2.c
tools/xngen src/f32-sigmoid/scalar-p5-div.c.in -D BATCH_TILE=4 -o src/f32-sigmoid/gen/scalar-p5-div-x4.c

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-sigmoid.yaml --output test/f32-sigmoid.cc
