#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-vlog/scalar-log.c.in -D BATCH_TILES=1,2,4 -o src/f32-vlog/gen/f32-vlog-scalar-log.c &

wait
