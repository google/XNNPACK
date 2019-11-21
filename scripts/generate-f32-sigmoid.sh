#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-sigmoid/neonfma-p5-nr2fma.c.in -D BATCH_TILE=16 -o src/f32-sigmoid/neonfma-p5-nr2fma-x16.c
tools/xngen src/f32-sigmoid/neon-frac-p9-p10-nr1recps.c.in -D BATCH_TILE=16 -o src/f32-sigmoid/neon-frac-p9-p10-nr1recps-x16.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=8 -D BLEND=0 -o src/f32-sigmoid/sse2-p5-div-x8.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=16 -D BLEND=0 -o src/f32-sigmoid/sse2-p5-div-x16.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=8 -D BLEND=1 -o src/f32-sigmoid/sse41-p5-div-x8.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=16 -D BLEND=1 -o src/f32-sigmoid/sse41-p5-div-x16.c

################################## Unit tests #################################
tools/generate-vunop-test.py --spec test/f32-sigmoid.yaml --output test/f32-sigmoid.cc
