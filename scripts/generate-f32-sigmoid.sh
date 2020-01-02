#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=4  -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-p5-div-x4.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=8  -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-p5-div-x8.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=12 -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-p5-div-x12.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=16 -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-p5-div-x16.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=20 -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-p5-div-x20.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=24 -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-p5-div-x24.c

tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=4  -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-p5-nr2fma-x4.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=8  -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-p5-nr2fma-x8.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=12 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-p5-nr2fma-x12.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=16 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-p5-nr2fma-x16.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=20 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-p5-nr2fma-x20.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=24 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-p5-nr2fma-x24.c

tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=4  -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-p5-nr1recps1fma-x4.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=8  -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-p5-nr1recps1fma-x8.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=12 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-p5-nr1recps1fma-x12.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=16 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-p5-nr1recps1fma-x16.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=20 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-p5-nr1recps1fma-x20.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=24 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-p5-nr1recps1fma-x24.c

tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=4  -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-p5-nr2recps-x4.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=8  -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-p5-nr2recps-x8.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=12 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-p5-nr2recps-x12.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=16 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-p5-nr2recps-x16.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=20 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-p5-nr2recps-x20.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=24 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-p5-nr2recps-x24.c

tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=4  -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-p5-nr2recps-x4.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=8  -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-p5-nr2recps-x8.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=12 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-p5-nr2recps-x12.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=16 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-p5-nr2recps-x16.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=20 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-p5-nr2recps-x20.c
tools/xngen src/f32-sigmoid/neon-p5.c.in -D BATCH_TILE=24 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-p5-nr2recps-x24.c

tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-lut2048-p1-div-x4.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-lut2048-p1-div-x8.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-lut2048-p1-div-x12.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-lut2048-p1-div-x16.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-lut2048-p1-div-x20.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D FMA=1 -D DIV_ALGO=div -o src/f32-sigmoid/gen/neonfma-lut2048-p1-div-x24.c

tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2fma-x4.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2fma-x8.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2fma-x12.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2fma-x16.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2fma-x20.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2fma-x24.c

tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr1recps1fma-x4.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr1recps1fma-x8.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr1recps1fma-x12.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr1recps1fma-x16.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr1recps1fma-x20.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr1recps1fma-x24.c

tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2recps-x4.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2recps-x8.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2recps-x12.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2recps-x16.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2recps-x20.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neonfma-lut2048-p1-nr2recps-x24.c

tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-lut2048-p1-nr2recps-x4.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-lut2048-p1-nr2recps-x8.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-lut2048-p1-nr2recps-x12.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-lut2048-p1-nr2recps-x16.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-lut2048-p1-nr2recps-x20.c
tools/xngen src/f32-sigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-sigmoid/gen/neon-lut2048-p1-nr2recps-x24.c

tools/xngen src/f32-sigmoid/neon-frac-p9-p10-nr1recps.c.in -D BATCH_TILE=16 -o src/f32-sigmoid/gen/neon-frac-p9-p10-nr1recps-x16.c

################################### x86 SSE ###################################
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=4  -D BLEND=0 -o src/f32-sigmoid/gen/sse2-p5-div-x4.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=8  -D BLEND=0 -o src/f32-sigmoid/gen/sse2-p5-div-x8.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=12 -D BLEND=0 -o src/f32-sigmoid/gen/sse2-p5-div-x12.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=16 -D BLEND=0 -o src/f32-sigmoid/gen/sse2-p5-div-x16.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=20 -D BLEND=0 -o src/f32-sigmoid/gen/sse2-p5-div-x20.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=24 -D BLEND=0 -o src/f32-sigmoid/gen/sse2-p5-div-x24.c

tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=4  -D BLEND=1 -o src/f32-sigmoid/gen/sse41-p5-div-x4.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=8  -D BLEND=1 -o src/f32-sigmoid/gen/sse41-p5-div-x8.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=12 -D BLEND=1 -o src/f32-sigmoid/gen/sse41-p5-div-x12.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=16 -D BLEND=1 -o src/f32-sigmoid/gen/sse41-p5-div-x16.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=20 -D BLEND=1 -o src/f32-sigmoid/gen/sse41-p5-div-x20.c
tools/xngen src/f32-sigmoid/sse-p5-div.c.in -D BATCH_TILE=24 -D BLEND=1 -o src/f32-sigmoid/gen/sse41-p5-div-x24.c

#################################### PSIMD ####################################
tools/xngen src/f32-sigmoid/psimd-p5-div.c.in -D BATCH_TILE=4 -D BLEND=0 -o src/f32-sigmoid/gen/psimd-p5-div-x4.c
tools/xngen src/f32-sigmoid/psimd-p5-div.c.in -D BATCH_TILE=8 -D BLEND=0 -o src/f32-sigmoid/gen/psimd-p5-div-x8.c
tools/xngen src/f32-sigmoid/psimd-p5-div.c.in -D BATCH_TILE=12 -D BLEND=0 -o src/f32-sigmoid/gen/psimd-p5-div-x12.c
tools/xngen src/f32-sigmoid/psimd-p5-div.c.in -D BATCH_TILE=16 -D BLEND=0 -o src/f32-sigmoid/gen/psimd-p5-div-x16.c
tools/xngen src/f32-sigmoid/psimd-p5-div.c.in -D BATCH_TILE=20 -D BLEND=0 -o src/f32-sigmoid/gen/psimd-p5-div-x20.c
tools/xngen src/f32-sigmoid/psimd-p5-div.c.in -D BATCH_TILE=24 -D BLEND=0 -o src/f32-sigmoid/gen/psimd-p5-div-x24.c

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
