#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### SCALAR ###################################
tools/xngen src/s16-rmaxabs/scalar.c.in -D BATCH_TILE=1 -o src/s16-rmaxabs/gen/s16-rmaxabs-scalar-x1.c &
tools/xngen src/s16-rmaxabs/scalar.c.in -D BATCH_TILE=2 -o src/s16-rmaxabs/gen/s16-rmaxabs-scalar-x2.c &
tools/xngen src/s16-rmaxabs/scalar.c.in -D BATCH_TILE=3 -o src/s16-rmaxabs/gen/s16-rmaxabs-scalar-x3.c &
tools/xngen src/s16-rmaxabs/scalar.c.in -D BATCH_TILE=4 -o src/s16-rmaxabs/gen/s16-rmaxabs-scalar-x4.c &

################################### NEON ###################################
tools/xngen src/s16-rmaxabs/neon.c.in -D BATCH_TILE=8  -o src/s16-rmaxabs/gen/s16-rmaxabs-neon-x8.c &
tools/xngen src/s16-rmaxabs/neon.c.in -D BATCH_TILE=16 -o src/s16-rmaxabs/gen/s16-rmaxabs-neon-x16.c &
tools/xngen src/s16-rmaxabs/neon.c.in -D BATCH_TILE=24 -o src/s16-rmaxabs/gen/s16-rmaxabs-neon-x24.c &
tools/xngen src/s16-rmaxabs/neon.c.in -D BATCH_TILE=32 -o src/s16-rmaxabs/gen/s16-rmaxabs-neon-x32.c &

wait
