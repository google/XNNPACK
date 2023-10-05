#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### SCALAR ###################################
tools/xngen src/u32-vlog/scalar.c.in -D BATCH_TILE=1 -o src/u32-vlog/gen/u32-vlog-scalar-x1.c &
tools/xngen src/u32-vlog/scalar.c.in -D BATCH_TILE=2 -o src/u32-vlog/gen/u32-vlog-scalar-x2.c &
tools/xngen src/u32-vlog/scalar.c.in -D BATCH_TILE=3 -o src/u32-vlog/gen/u32-vlog-scalar-x3.c &
tools/xngen src/u32-vlog/scalar.c.in -D BATCH_TILE=4 -o src/u32-vlog/gen/u32-vlog-scalar-x4.c &

wait
