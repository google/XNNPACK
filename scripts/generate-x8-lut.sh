#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=1  -o src/x8-lut/gen/lut-scalar-x1.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=2  -o src/x8-lut/gen/lut-scalar-x2.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=4  -o src/x8-lut/gen/lut-scalar-x4.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=8  -o src/x8-lut/gen/lut-scalar-x8.c &
tools/xngen src/x8-lut/scalar.c.in -D BATCH_TILE=16 -o src/x8-lut/gen/lut-scalar-x16.c &

################################## Unit tests #################################
tools/generate-lut-test.py --spec test/x8-lut.yaml --output test/x8-lut.cc &

wait
