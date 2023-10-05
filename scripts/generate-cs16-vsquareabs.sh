#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ###################################
tools/xngen src/cs16-vsquareabs/scalar.c.in -D BATCH_TILE=1 -o src/cs16-vsquareabs/gen/cs16-vsquareabs-scalar-x1.c &
tools/xngen src/cs16-vsquareabs/scalar.c.in -D BATCH_TILE=2 -o src/cs16-vsquareabs/gen/cs16-vsquareabs-scalar-x2.c &
tools/xngen src/cs16-vsquareabs/scalar.c.in -D BATCH_TILE=3 -o src/cs16-vsquareabs/gen/cs16-vsquareabs-scalar-x3.c &
tools/xngen src/cs16-vsquareabs/scalar.c.in -D BATCH_TILE=4 -o src/cs16-vsquareabs/gen/cs16-vsquareabs-scalar-x4.c &

################################### NEON ###################################
tools/xngen src/cs16-vsquareabs/neon.c.in -D BATCH_TILE=4  -o src/cs16-vsquareabs/gen/cs16-vsquareabs-neon-mlal-ld128-x4.c &
tools/xngen src/cs16-vsquareabs/neon.c.in -D BATCH_TILE=8  -o src/cs16-vsquareabs/gen/cs16-vsquareabs-neon-mlal-ld128-x8.c &
tools/xngen src/cs16-vsquareabs/neon.c.in -D BATCH_TILE=12 -o src/cs16-vsquareabs/gen/cs16-vsquareabs-neon-mlal-ld128-x12.c &
tools/xngen src/cs16-vsquareabs/neon.c.in -D BATCH_TILE=16 -o src/cs16-vsquareabs/gen/cs16-vsquareabs-neon-mlal-ld128-x16.c &

################################### Hexagon ###################################
tools/xngen src/cs16-vsquareabs/hexagon.c.in -D BATCH_TILE=2  -o src/cs16-vsquareabs/gen/cs16-vsquareabs-hexagon-x2.c &
tools/xngen src/cs16-vsquareabs/hexagon.c.in -D BATCH_TILE=4  -o src/cs16-vsquareabs/gen/cs16-vsquareabs-hexagon-x4.c &
tools/xngen src/cs16-vsquareabs/hexagon.c.in -D BATCH_TILE=6  -o src/cs16-vsquareabs/gen/cs16-vsquareabs-hexagon-x6.c &
tools/xngen src/cs16-vsquareabs/hexagon.c.in -D BATCH_TILE=8  -o src/cs16-vsquareabs/gen/cs16-vsquareabs-hexagon-x8.c &
tools/xngen src/cs16-vsquareabs/hexagon.c.in -D BATCH_TILE=10 -o src/cs16-vsquareabs/gen/cs16-vsquareabs-hexagon-x10.c &
tools/xngen src/cs16-vsquareabs/hexagon.c.in -D BATCH_TILE=12 -o src/cs16-vsquareabs/gen/cs16-vsquareabs-hexagon-x12.c &

wait
