#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-p5-div-u4.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-p5-div-u8.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-p5-div-u12.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-p5-div-u16.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-p5-div-u20.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-p5-div-u24.c &

tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2fma-u4.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2fma-u8.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2fma-u12.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2fma-u16.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2fma-u20.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2fma-u24.c &

tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr1recps1fma-u4.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr1recps1fma-u8.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr1recps1fma-u12.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr1recps1fma-u16.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr1recps1fma-u20.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr1recps1fma-u24.c &

tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2recps-u4.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2recps-u8.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2recps-u12.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2recps-u16.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2recps-u20.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-p5-nr2recps-u24.c &

tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=4  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-p5-nr2recps-u4.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=8  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-p5-nr2recps-u8.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=12 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-p5-nr2recps-u12.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=16 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-p5-nr2recps-u16.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=20 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-p5-nr2recps-u20.c &
tools/xngen src/f32-vsigmoid/neon-p5.c.in -D BATCH_TILE=24 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-p5-nr2recps-u24.c &

tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut64-p2-div-u4.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut64-p2-div-u8.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut64-p2-div-u12.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut64-p2-div-u16.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut64-p2-div-u20.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut64-p2-div-u24.c &

tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2fma-u4.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2fma-u8.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2fma-u12.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2fma-u16.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2fma-u20.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2fma-u24.c &

tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-u4.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-u8.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-u12.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-u16.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-u20.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr1recps1fma-u24.c &

tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2recps-u4.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2recps-u8.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2recps-u12.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2recps-u16.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2recps-u20.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut64-p2-nr2recps-u24.c &

tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=4  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut64-p2-nr2recps-u4.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=8  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut64-p2-nr2recps-u8.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=12 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut64-p2-nr2recps-u12.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=16 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut64-p2-nr2recps-u16.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=20 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut64-p2-nr2recps-u20.c &
tools/xngen src/f32-vsigmoid/neon-lut64-p2.c.in -D BATCH_TILE=24 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut64-p2-nr2recps-u24.c &

tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut2048-p1-div-u4.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut2048-p1-div-u8.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut2048-p1-div-u12.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut2048-p1-div-u16.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut2048-p1-div-u20.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-aarch64-neonfma-rr1-lut2048-p1-div-u24.c &

tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-u4.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-u8.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-u12.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-u16.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-u20.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2fma-u24.c &

tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-u4.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-u8.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-u12.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-u16.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-u20.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr1recps1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr1recps1fma-u24.c &

tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-u4.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-u8.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-u12.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-u16.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-u20.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D RR_STEPS=1 -D FMA=1 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neonfma-rr1-lut2048-p1-nr2recps-u24.c &

tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=4  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut2048-p1-nr2recps-u4.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=8  -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut2048-p1-nr2recps-u8.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=12 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut2048-p1-nr2recps-u12.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=16 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut2048-p1-nr2recps-u16.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=20 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut2048-p1-nr2recps-u20.c &
tools/xngen src/f32-vsigmoid/neon-lut2048-p1.c.in -D BATCH_TILE=24 -D RR_STEPS=2 -D FMA=0 -D DIV_ALGO=nr2recps -o src/f32-vsigmoid/gen/f32-vsigmoid-neon-rr2-lut2048-p1-nr2recps-u24.c &

################################### x86 SSE ###################################
tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=4  -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-p5-div-u4.c &
tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=8  -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-p5-div-u8.c &
tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=12 -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-p5-div-u12.c &
tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=16 -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-p5-div-u16.c &
tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=20 -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-p5-div-u20.c &
tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=24 -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-p5-div-u24.c &

tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=4  -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-p5-div-u4.c &
tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=8  -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-p5-div-u8.c &
tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=12 -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-p5-div-u12.c &
tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=16 -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-p5-div-u16.c &
tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=20 -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-p5-div-u20.c &
tools/xngen src/f32-vsigmoid/sse-rr2-p5-div.c.in -D BATCH_TILE=24 -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-p5-div-u24.c &

tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=4  -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-lut64-p2-div-u4.c &
tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=8  -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-lut64-p2-div-u8.c &
tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=12 -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-lut64-p2-div-u12.c &
tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=16 -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-lut64-p2-div-u16.c &
tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=20 -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-lut64-p2-div-u20.c &
tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=24 -D SSE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse2-rr2-lut64-p2-div-u24.c &

tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=4  -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-lut64-p2-div-u4.c &
tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=8  -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-lut64-p2-div-u8.c &
tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=12 -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-lut64-p2-div-u12.c &
tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=16 -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-lut64-p2-div-u16.c &
tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=20 -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-lut64-p2-div-u20.c &
tools/xngen src/f32-vsigmoid/sse-rr2-lut64-p2-div.c.in -D BATCH_TILE=24 -D SSE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-sse41-rr2-lut64-p2-div-u24.c &

################################### x86 AVX ###################################
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=8  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u8.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=16 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u16.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=24 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u24.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=32 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u32.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=40 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u40.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=48 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u48.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=56 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u56.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=64 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u64.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=72 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u72.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=80 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-div-u80.c &

tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=8  -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u8.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=16 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u16.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=24 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u24.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=32 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u32.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=40 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u40.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=48 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u48.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=56 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u56.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=64 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u64.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=72 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u72.c &
tools/xngen src/f32-vsigmoid/avx-rr2-p5.c.in -D BATCH_TILE=80 -D DIV_ALGO=nr2 -o src/f32-vsigmoid/gen/f32-vsigmoid-avx-rr2-p5-nr2-u80.c &

tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=8  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u8.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=16 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u16.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=24 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u24.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=32 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u32.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=40 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u40.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=48 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u48.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=56 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u56.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=64 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u64.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=72 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u72.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=80 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-div-u80.c &

tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=8  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u8.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=16 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u16.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=24 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u24.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=32 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u32.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=40 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u40.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=48 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u48.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=56 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u56.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=64 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u64.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=72 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u72.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=80 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr1fma-u80.c &

tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=8  -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u8.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=16 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u16.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=24 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u24.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=32 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u32.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=40 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u40.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=48 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u48.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=56 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u56.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=64 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u64.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=72 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u72.c &
tools/xngen src/f32-vsigmoid/avx2-rr1-p5.c.in -D BATCH_TILE=80 -D DIV_ALGO=nr2fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx2-rr1-p5-nr2fma-u80.c &

################################# x86 AVX-512 #################################
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=16  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-div-u16.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=32  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-div-u32.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=48  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-div-u48.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=64  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-div-u64.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=80  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-div-u80.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=96  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-div-u96.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=112 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-div-u112.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=128 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-div-u128.c &

tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=16  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-nr1fma-u16.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=32  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-nr1fma-u32.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=48  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-nr1fma-u48.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=64  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-nr1fma-u64.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=80  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-nr1fma-u80.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=96  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-nr1fma-u96.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=112 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-nr1fma-u112.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-p5-scalef.c.in -D BATCH_TILE=128 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-p5-scalef-nr1fma-u128.c &

tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=16  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-u16.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=32  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-u32.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=48  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-u48.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=64  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-u64.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=80  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-u80.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=96  -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-u96.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=112 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-u112.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=128 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-div-u128.c &

tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=16  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-u16.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=32  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-u32.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=48  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-u48.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=64  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-u64.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=80  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-u80.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=96  -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-u96.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=112 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-u112.c &
tools/xngen src/f32-vsigmoid/avx512f-rr1-lut16-p3-perm-scalef.c.in -D BATCH_TILE=128 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr1-lut16-p3-perm-scalef-nr1fma-u128.c &

tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=16  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-u16.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=32  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-u32.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=48  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-u48.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=64  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-u64.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=80  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-u80.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=96  -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-u96.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=112 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-u112.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=128 -D RR_STEPS=2 -D DIV_ALGO=div -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-div-u128.c &

tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=16  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-u16.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=32  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-u32.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=48  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-u48.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=64  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-u64.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=80  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-u80.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=96  -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-u96.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=112 -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-u112.c &
tools/xngen src/f32-vsigmoid/avx512f-rr2-lut32-p2-perm2-scalef.c.in -D BATCH_TILE=128 -D RR_STEPS=2 -D DIV_ALGO=nr1fma -o src/f32-vsigmoid/gen/f32-vsigmoid-avx512f-rr2-lut32-p2-perm2-scalef-nr1fma-u128.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=4  -D RELAXED=0 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-p5-div-u4.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=8  -D RELAXED=0 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-p5-div-u8.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=12 -D RELAXED=0 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-p5-div-u12.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=16 -D RELAXED=0 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-p5-div-u16.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=20 -D RELAXED=0 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-p5-div-u20.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=24 -D RELAXED=0 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-p5-div-u24.c &

tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=4  -D RELAXED=1 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u4.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=8  -D RELAXED=1 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u8.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=12 -D RELAXED=1 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u12.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=16 -D RELAXED=1 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u16.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=20 -D RELAXED=1 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u20.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=24 -D RELAXED=1 -D FMA=0 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-p5-div-u24.c &

tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=4  -D RELAXED=1 -D FMA=1 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u4.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=8  -D RELAXED=1 -D FMA=1 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u8.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=12 -D RELAXED=1 -D FMA=1 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u12.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=16 -D RELAXED=1 -D FMA=1 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u16.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=20 -D RELAXED=1 -D FMA=1 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u20.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=24 -D RELAXED=1 -D FMA=1 -D BLENDVPS=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-p5-div-u24.c &

tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=4  -D RELAXED=1 -D FMA=0 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u4.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=8  -D RELAXED=1 -D FMA=0 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u8.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=12 -D RELAXED=1 -D FMA=0 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u12.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=16 -D RELAXED=1 -D FMA=0 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u16.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=20 -D RELAXED=1 -D FMA=0 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u20.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=24 -D RELAXED=1 -D FMA=0 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-rr2-p5-div-u24.c &

tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=4  -D RELAXED=1 -D FMA=1 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u4.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=8  -D RELAXED=1 -D FMA=1 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u8.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=12 -D RELAXED=1 -D FMA=1 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u12.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=16 -D RELAXED=1 -D FMA=1 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u16.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=20 -D RELAXED=1 -D FMA=1 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u20.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-p5-div.c.in -D BATCH_TILE=24 -D RELAXED=1 -D FMA=1 -D BLENDVPS=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmblendvps-fma-rr2-p5-div-u24.c &

tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=4  -D RELAXED=0 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-lut64-p2-div-u4.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=8  -D RELAXED=0 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-lut64-p2-div-u8.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=12 -D RELAXED=0 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-lut64-p2-div-u12.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=16 -D RELAXED=0 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-lut64-p2-div-u16.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=20 -D RELAXED=0 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-lut64-p2-div-u20.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=24 -D RELAXED=0 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmsimd-rr2-lut64-p2-div-u24.c &

tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=4  -D RELAXED=1 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u4.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=8  -D RELAXED=1 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u8.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=12 -D RELAXED=1 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u12.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=16 -D RELAXED=1 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u16.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=20 -D RELAXED=1 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u20.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=24 -D RELAXED=1 -D FMA=0 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-rr2-lut64-p2-div-u24.c &

tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=4  -D RELAXED=1 -D FMA=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u4.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=8  -D RELAXED=1 -D FMA=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u8.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=12 -D RELAXED=1 -D FMA=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u12.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=16 -D RELAXED=1 -D FMA=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u16.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=20 -D RELAXED=1 -D FMA=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u20.c &
tools/xngen src/f32-vsigmoid/wasmsimd-rr2-lut64-p2-div.c.in -D BATCH_TILE=24 -D RELAXED=1 -D FMA=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-wasmrelaxedsimd-fma-rr2-lut64-p2-div-u24.c &

################################### Scalar ####################################
tools/xngen src/f32-vsigmoid/scalar-rr2-lut2048-p1-div.c.in -D BATCH_TILE=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-scalar-rr2-lut2048-p1-div-u1.c &
tools/xngen src/f32-vsigmoid/scalar-rr2-lut2048-p1-div.c.in -D BATCH_TILE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-scalar-rr2-lut2048-p1-div-u2.c &
tools/xngen src/f32-vsigmoid/scalar-rr2-lut2048-p1-div.c.in -D BATCH_TILE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-scalar-rr2-lut2048-p1-div-u4.c &

tools/xngen src/f32-vsigmoid/scalar-rr2-lut64-p2-div.c.in -D BATCH_TILE=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-scalar-rr2-lut64-p2-div-u1.c &
tools/xngen src/f32-vsigmoid/scalar-rr2-lut64-p2-div.c.in -D BATCH_TILE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-scalar-rr2-lut64-p2-div-u2.c &
tools/xngen src/f32-vsigmoid/scalar-rr2-lut64-p2-div.c.in -D BATCH_TILE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-scalar-rr2-lut64-p2-div-u4.c &

tools/xngen src/f32-vsigmoid/scalar-rr2-p5-div.c.in -D BATCH_TILE=1 -o src/f32-vsigmoid/gen/f32-vsigmoid-scalar-rr2-p5-div-u1.c &
tools/xngen src/f32-vsigmoid/scalar-rr2-p5-div.c.in -D BATCH_TILE=2 -o src/f32-vsigmoid/gen/f32-vsigmoid-scalar-rr2-p5-div-u2.c &
tools/xngen src/f32-vsigmoid/scalar-rr2-p5-div.c.in -D BATCH_TILE=4 -o src/f32-vsigmoid/gen/f32-vsigmoid-scalar-rr2-p5-div-u4.c &

wait
