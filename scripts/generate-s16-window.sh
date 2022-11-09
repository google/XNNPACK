#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### SCALAR ###################################
tools/xngen src/s16-window/scalar.c.in -D CHANNEL_TILE=1 -o src/s16-window/gen/s16-window-scalar-x1.c &
tools/xngen src/s16-window/scalar.c.in -D CHANNEL_TILE=2 -o src/s16-window/gen/s16-window-scalar-x2.c &
tools/xngen src/s16-window/scalar.c.in -D CHANNEL_TILE=3 -o src/s16-window/gen/s16-window-scalar-x3.c &
tools/xngen src/s16-window/scalar.c.in -D CHANNEL_TILE=4 -o src/s16-window/gen/s16-window-scalar-x4.c &

################################### NEON ###################################
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=8  -D SHIFT=0   -o src/s16-window/gen/s16-window-neon-x8.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=16 -D SHIFT=0   -o src/s16-window/gen/s16-window-neon-x16.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=24 -D SHIFT=0   -o src/s16-window/gen/s16-window-neon-x24.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=32 -D SHIFT=0   -o src/s16-window/gen/s16-window-neon-x32.c &

tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=8  -D SHIFT=12  -o src/s16-window/gen/s16-window-shift12-neon-x8.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=16 -D SHIFT=12  -o src/s16-window/gen/s16-window-shift12-neon-x16.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=24 -D SHIFT=12  -o src/s16-window/gen/s16-window-shift12-neon-x24.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=32 -D SHIFT=12  -o src/s16-window/gen/s16-window-shift12-neon-x32.c &

tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=8  -D SHIFT=15  -o src/s16-window/gen/s16-window-shift15-neon-x8.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=16 -D SHIFT=15  -o src/s16-window/gen/s16-window-shift15-neon-x16.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=24 -D SHIFT=15  -o src/s16-window/gen/s16-window-shift15-neon-x24.c &
tools/xngen src/s16-window/neon.c.in -D CHANNEL_TILE=32 -D SHIFT=15  -o src/s16-window/gen/s16-window-shift15-neon-x32.c &

################################## Unit tests #################################
tools/generate-window-test.py --spec test/s16-window.yaml --output test/s16-window.cc &

wait
