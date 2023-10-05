#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### SCALAR ###################################
tools/xngen src/u32-filterbank-accumulate/scalar.c.in -D BATCH_TILE=1 -o src/u32-filterbank-accumulate/gen/u32-filterbank-accumulate-scalar-x1.c &

################################### NEON ###################################
tools/xngen src/u32-filterbank-accumulate/neon.c.in -D BATCH_TILE=1 -o src/u32-filterbank-accumulate/gen/u32-filterbank-accumulate-neon-x1.c &
tools/xngen src/u32-filterbank-accumulate/neon.c.in -D BATCH_TILE=2 -o src/u32-filterbank-accumulate/gen/u32-filterbank-accumulate-neon-x2.c &

wait
