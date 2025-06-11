#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f16-vapproxgelu/rational-6-4.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8 -o src/f16-vapproxgelu/gen/f16-vapproxgelu-scalar-rational-6-4-div.c &
tools/xngen src/f16-vapproxgelu/rational-6-4.c.in -D ARCH=neonfp16arith -D BATCH_TILES=8,16,32 -o src/f16-vapproxgelu/gen/f16-vapproxgelu-neonfp16arith-rational-6-4-div.c &

wait
