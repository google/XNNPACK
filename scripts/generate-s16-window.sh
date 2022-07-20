#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### SCALAR ###################################
tools/xngen src/s16-window/scalar.c.in -D CHANNEL_TILE=1 -o src/s16-window/gen/scalar-x1.c &
tools/xngen src/s16-window/scalar.c.in -D CHANNEL_TILE=2 -o src/s16-window/gen/scalar-x2.c &
tools/xngen src/s16-window/scalar.c.in -D CHANNEL_TILE=4 -o src/s16-window/gen/scalar-x4.c &

################################## Unit tests #################################
tools/generate-window-test.py --spec test/s16-window.yaml --output test/s16-window.cc &

wait
