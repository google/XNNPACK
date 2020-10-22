#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-ibilinear-chw/scalar.c.in -D PIXEL_TILE=1 -o src/f32-ibilinear-chw/gen/scalar-p1.c
tools/xngen src/f32-ibilinear-chw/scalar.c.in -D PIXEL_TILE=2 -o src/f32-ibilinear-chw/gen/scalar-p2.c
tools/xngen src/f32-ibilinear-chw/scalar.c.in -D PIXEL_TILE=4 -o src/f32-ibilinear-chw/gen/scalar-p4.c
