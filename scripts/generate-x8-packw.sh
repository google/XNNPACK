#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x32-packw/scalar.c.in -D NR=2 -D KUNROLL=2 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x2-scalar-int-x2.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=4 -D KUNROLL=4 -D TYPE=int8_t -o src/x8-packw/gen/x8-packw-x4-scalar-int-x4.c &

################################## Unit tests #################################
tools/generate-packw-test.py --spec test/x8-packw.yaml --output test/x8-packw.cc &

wait
