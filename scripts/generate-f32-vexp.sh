#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-vexp/scalar-exp.c.in -D BATCH_TILES=1,2,4 -o src/f32-vexp/gen/f32-vexp-scalar-exp.c &

wait
