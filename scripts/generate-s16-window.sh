#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### SCALAR ###################################
tools/xngen src/s16-window/scalar.c.in -D CHANNEL_TILE=1 -o src/s16-window/gen/s16-window-scalar-u1.c &
tools/xngen src/s16-window/scalar.c.in -D CHANNEL_TILE=2 -o src/s16-window/gen/s16-window-scalar-u2.c &
tools/xngen src/s16-window/scalar.c.in -D CHANNEL_TILE=3 -o src/s16-window/gen/s16-window-scalar-u3.c &
tools/xngen src/s16-window/scalar.c.in -D CHANNEL_TILE=4 -o src/s16-window/gen/s16-window-scalar-u4.c &

################################### NEON ###################################
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=8  -D SHIFT=0   -o src/s16-window/gen/s16-window-neon-u8.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=16 -D SHIFT=0   -o src/s16-window/gen/s16-window-neon-u16.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=24 -D SHIFT=0   -o src/s16-window/gen/s16-window-neon-u24.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=32 -D SHIFT=0   -o src/s16-window/gen/s16-window-neon-u32.c &

tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=8  -D SHIFT=12  -o src/s16-window/gen/s16-window-shift12-neon-u8.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=16 -D SHIFT=12  -o src/s16-window/gen/s16-window-shift12-neon-u16.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=24 -D SHIFT=12  -o src/s16-window/gen/s16-window-shift12-neon-u24.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=32 -D SHIFT=12  -o src/s16-window/gen/s16-window-shift12-neon-u32.c &

tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=8  -D SHIFT=15  -o src/s16-window/gen/s16-window-shift15-neon-u8.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=16 -D SHIFT=15  -o src/s16-window/gen/s16-window-shift15-neon-u16.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=24 -D SHIFT=15  -o src/s16-window/gen/s16-window-shift15-neon-u24.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=32 -D SHIFT=15  -o src/s16-window/gen/s16-window-shift15-neon-u32.c &

wait
