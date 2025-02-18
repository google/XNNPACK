#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f16-vgelu/rational-6-4.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8 -o src/f16-vgelu/gen/f16-vgelu-scalar-rational-6-4-div.c &

wait
