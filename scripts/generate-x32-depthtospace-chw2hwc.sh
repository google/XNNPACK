#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/x32-depth-to-space-chw2hwc/scalar.c.in -D CHANNEL_TILE=1 -D INNER_BLOCK_SIZE_TILE=1 -o src/x32-depth-to-space-chw2hwc/gen/scalar-c1.c
tools/xngen src/x32-depth-to-space-chw2hwc/scalar.c.in -D CHANNEL_TILE=2 -D INNER_BLOCK_SIZE_TILE=1 -o src/x32-depth-to-space-chw2hwc/gen/scalar-c2.c
tools/xngen src/x32-depth-to-space-chw2hwc/scalar.c.in -D CHANNEL_TILE=4 -D INNER_BLOCK_SIZE_TILE=1 -o src/x32-depth-to-space-chw2hwc/gen/scalar-c4.c
tools/xngen src/x32-depth-to-space-chw2hwc/scalar.c.in -D CHANNEL_TILE=1 -D INNER_BLOCK_SIZE_TILE=2 -o src/x32-depth-to-space-chw2hwc/gen/scalar-c1-ib2.c
tools/xngen src/x32-depth-to-space-chw2hwc/scalar.c.in -D CHANNEL_TILE=2 -D INNER_BLOCK_SIZE_TILE=2 -o src/x32-depth-to-space-chw2hwc/gen/scalar-c2-ib2.c
tools/xngen src/x32-depth-to-space-chw2hwc/scalar.c.in -D CHANNEL_TILE=4 -D INNER_BLOCK_SIZE_TILE=2 -o src/x32-depth-to-space-chw2hwc/gen/scalar-c4-ib2.c

################################## Unit tests #################################
tools/generate-depthtospace-chw2hwc-test.py --spec test/x32-depthtospace-chw2hwc.yaml --output test/x32-depthtospace-chw2hwc.cc
