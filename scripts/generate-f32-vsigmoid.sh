#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x4.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x8.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x12.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x16.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x20.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-div-x24.c

tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x4.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x8.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x12.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x16.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x20.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2fma-x24.c

tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x4.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x8.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x12.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x16.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x20.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr1recps1fma-x24.c

tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x4.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x8.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x12.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x16.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x20.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-p5-nr2recps-x24.c

tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=4  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x4.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x8.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=12 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x12.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x16.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=20 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x20.c
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-p5-nr2recps-x24.c

tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x4.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x8.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x12.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x16.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x20.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-div-x24.c

tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x4.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x8.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x12.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x16.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x20.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2fma-x24.c

tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x4.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x8.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x12.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x16.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x20.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-x24.c

tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x4.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x8.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x12.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x16.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x20.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut64-p2-nr2recps-x24.c

tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=4  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x4.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=8  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x8.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=12 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x12.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=16 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x16.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=20 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x20.c
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=24 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut64-p2-nr2recps-x24.c

tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x4.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x8.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x12.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x16.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x20.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-div-x24.c

tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x4.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x8.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x12.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x16.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x20.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-x24.c

tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x4.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x8.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x12.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x16.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x20.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-x24.c

tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x4.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x8.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x12.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x16.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x20.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-x24.c

tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x4.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x8.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x12.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x16.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x20.c
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/vsigmoid-neon-rr2-lut2048-p1-nr2recps-x24.c

################################### x86 SSE ###################################
tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=4  -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x4.c
tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=8  -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x8.c
tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=12 -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x12.c
tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=16 -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x16.c
tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=20 -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x20.c
tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=24 -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-p5-div-x24.c

tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=4  -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x4.c
tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=8  -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x8.c
tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=12 -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x12.c
tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=16 -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x16.c
tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=20 -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x20.c
tools/xngen src/f32-vsigmoid/sse-p5-div.c.in -D BATCH_TILE=24 -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-p5-div-x24.c

tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=4  -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x4.c
tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=8  -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x8.c
tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=12 -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x12.c
tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=16 -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x16.c
tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=20 -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x20.c
tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=24 -D SSE=2 -o src/f32-vsigmoid/gen/vsigmoid-sse2-lut64-p2-div-x24.c

tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=4  -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x4.c
tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=8  -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x8.c
tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=12 -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x12.c
tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=16 -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x16.c
tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=20 -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x20.c
tools/xngen src/f32-vsigmoid/sse-lut64-p2-div.c.in -D BATCH_TILE=24 -D SSE=4 -o src/f32-vsigmoid/gen/vsigmoid-sse41-lut64-p2-div-x24.c

################################### x86 AVX ###################################
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x8.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x16.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x24.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=32 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x32.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=40 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x40.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=48 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x48.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=56 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x56.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=64 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x64.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=72 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x72.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=80 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-div-x80.c

tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=2 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x8.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=2 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x16.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=2 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x24.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=32 -D RR_STEPS=2 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x32.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=40 -D RR_STEPS=2 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x40.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=48 -D RR_STEPS=2 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x48.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=56 -D RR_STEPS=2 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x56.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=64 -D RR_STEPS=2 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x64.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=72 -D RR_STEPS=2 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x72.c
tools/xngen src/f32-vsigmoid/avx-p5.c.in -D BATCH_TILE=80 -D RR_STEPS=2 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/vsigmoid-avx-rr2-p5-nr2-x80.c

tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x8.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x16.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x24.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=32 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x32.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=40 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x40.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=48 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x48.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=56 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x56.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=64 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x64.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=72 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x72.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=80 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-div-x80.c

tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x8.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x16.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x24.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=32 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x32.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=40 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x40.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=48 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x48.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=56 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x56.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=64 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x64.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=72 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x72.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=80 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr1fma-x80.c

tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x8.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x16.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x24.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=32 -D RR_STEPS=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x32.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=40 -D RR_STEPS=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x40.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=48 -D RR_STEPS=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x48.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=56 -D RR_STEPS=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x56.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=64 -D RR_STEPS=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x64.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=72 -D RR_STEPS=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x72.c
tools/xngen src/f32-vsigmoid/avx2-p5.c.in -D BATCH_TILE=80 -D RR_STEPS=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/vsigmoid-avx2-rr1-p5-nr2fma-x80.c

################################# x86 AVX-512 #################################
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=16  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x16.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=32  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x32.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=48  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x48.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=64  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x64.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=80  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x80.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=96  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x96.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=112 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x112.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=128 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-div-x128.c

tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=16  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x16.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=32  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x32.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=48  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x48.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=64  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x64.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=80  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x80.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=96  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x96.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=112 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x112.c
tools/xngen src/f32-vsigmoid/avx512f-p5-scalef.c.in -D BATCH_TILE=128 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-p5-scalef-nr1fma-x128.c

tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=16  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x16.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=32  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x32.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=48  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x48.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=64  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x64.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=80  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x80.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=96  -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x96.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=112 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x112.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=128 -D RR_STEPS=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-x128.c

tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=16  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x16.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=32  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x32.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=48  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x48.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=64  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x64.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=80  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x80.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=96  -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x96.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=112 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x112.c
tools/xngen src/f32-vsigmoid/avx512f-lut16-p3-perm-scalef.c.in -D BATCH_TILE=128 -D RR_STEPS=1 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-x128.c

tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=16  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x16.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=32  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x32.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=48  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x48.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=64  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x64.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=80  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x80.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=96  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x96.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=112 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x112.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=128 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-x128.c

tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=16  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x16.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=32  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x32.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=48  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x48.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=64  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x64.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=80  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x80.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=96  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x96.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=112 -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x112.c
tools/xngen src/f32-vsigmoid/avx512f-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=128 -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-x128.c

################################## WAsm SIMD ##################################
tools/xngen src/f32-vsigmoid/wasmsimd-p5-div.c.in -D BATCH_TILE=4  -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-p5-div-x4.c
tools/xngen src/f32-vsigmoid/wasmsimd-p5-div.c.in -D BATCH_TILE=8  -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-p5-div-x8.c
tools/xngen src/f32-vsigmoid/wasmsimd-p5-div.c.in -D BATCH_TILE=12 -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-p5-div-x12.c
tools/xngen src/f32-vsigmoid/wasmsimd-p5-div.c.in -D BATCH_TILE=16 -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-p5-div-x16.c
tools/xngen src/f32-vsigmoid/wasmsimd-p5-div.c.in -D BATCH_TILE=20 -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-p5-div-x20.c
tools/xngen src/f32-vsigmoid/wasmsimd-p5-div.c.in -D BATCH_TILE=24 -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-p5-div-x24.c

tools/xngen src/f32-vsigmoid/wasmsimd-lut64-p2-div.c.in -D BATCH_TILE=4  -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-lut64-p2-div-x4.c
tools/xngen src/f32-vsigmoid/wasmsimd-lut64-p2-div.c.in -D BATCH_TILE=8  -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-lut64-p2-div-x8.c
tools/xngen src/f32-vsigmoid/wasmsimd-lut64-p2-div.c.in -D BATCH_TILE=12 -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-lut64-p2-div-x12.c
tools/xngen src/f32-vsigmoid/wasmsimd-lut64-p2-div.c.in -D BATCH_TILE=16 -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-lut64-p2-div-x16.c
tools/xngen src/f32-vsigmoid/wasmsimd-lut64-p2-div.c.in -D BATCH_TILE=20 -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-lut64-p2-div-x20.c
tools/xngen src/f32-vsigmoid/wasmsimd-lut64-p2-div.c.in -D BATCH_TILE=24 -D BLEND=0 -o src/f32-vsigmoid/gen/vsigmoid-wasmsimd-lut64-p2-div-x24.c

################################### Scalar ####################################
tools/xngen src/f32-vsigmoid/scalar-lut2048-p1-div.c.in -D BATCH_TILE=1 -o src/f32-vsigmoid/gen/vsigmoid-scalar-lut2048-p1-div-x1.c
tools/xngen src/f32-vsigmoid/scalar-lut2048-p1-div.c.in -D BATCH_TILE=2 -o src/f32-vsigmoid/gen/vsigmoid-scalar-lut2048-p1-div-x2.c
tools/xngen src/f32-vsigmoid/scalar-lut2048-p1-div.c.in -D BATCH_TILE=4 -o src/f32-vsigmoid/gen/vsigmoid-scalar-lut2048-p1-div-x4.c

tools/xngen src/f32-vsigmoid/scalar-lut64-p2-div.c.in -D BATCH_TILE=1 -o src/f32-vsigmoid/gen/vsigmoid-scalar-lut64-p2-div-x1.c
tools/xngen src/f32-vsigmoid/scalar-lut64-p2-div.c.in -D BATCH_TILE=2 -o src/f32-vsigmoid/gen/vsigmoid-scalar-lut64-p2-div-x2.c
tools/xngen src/f32-vsigmoid/scalar-lut64-p2-div.c.in -D BATCH_TILE=4 -o src/f32-vsigmoid/gen/vsigmoid-scalar-lut64-p2-div-x4.c

tools/xngen src/f32-vsigmoid/scalar-p5-div.c.in -D BATCH_TILE=1 -o src/f32-vsigmoid/gen/vsigmoid-scalar-p5-div-x1.c
tools/xngen src/f32-vsigmoid/scalar-p5-div.c.in -D BATCH_TILE=2 -o src/f32-vsigmoid/gen/vsigmoid-scalar-p5-div-x2.c
tools/xngen src/f32-vsigmoid/scalar-p5-div.c.in -D BATCH_TILE=4 -o src/f32-vsigmoid/gen/vsigmoid-scalar-p5-div-x4.c

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vsigmoid.yaml --output test/f32-vsigmoid.cc
