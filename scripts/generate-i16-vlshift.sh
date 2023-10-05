#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### SCALAR ###################################
tools/xngen src/i16-vlshift/scalar.c.in -D BATCH_TILE=1 -o src/i16-vlshift/gen/i16-vlshift-scalar-u1.c &
tools/xngen src/i16-vlshift/scalar.c.in -D BATCH_TILE=2 -o src/i16-vlshift/gen/i16-vlshift-scalar-u2.c &
tools/xngen src/i16-vlshift/scalar.c.in -D BATCH_TILE=3 -o src/i16-vlshift/gen/i16-vlshift-scalar-u3.c &
tools/xngen src/i16-vlshift/scalar.c.in -D BATCH_TILE=4 -o src/i16-vlshift/gen/i16-vlshift-scalar-u4.c &

################################### NEON ###################################
tools/xngen src/i16-vlshift/neon.c.in -D BATCH_TILE=8  -o src/i16-vlshift/gen/i16-vlshift-neon-u8.c &
tools/xngen src/i16-vlshift/neon.c.in -D BATCH_TILE=16 -o src/i16-vlshift/gen/i16-vlshift-neon-u16.c &
tools/xngen src/i16-vlshift/neon.c.in -D BATCH_TILE=24 -o src/i16-vlshift/gen/i16-vlshift-neon-u24.c &
tools/xngen src/i16-vlshift/neon.c.in -D BATCH_TILE=32 -o src/i16-vlshift/gen/i16-vlshift-neon-u32.c &

wait
