#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x32-packw/scalar.c.in -D NR=8  -D KUNROLL=4 -D TYPE=uint16_t -o src/x16-packw/gen/x16-packw-x8-scalar-int.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=16 -D KUNROLL=4 -D TYPE=uint16_t -o src/x16-packw/gen/x16-packw-x16-scalar-int.c &

################################## Unit tests #################################
tools/generate-packw-test.py --spec test/x16-packw.yaml --output test/x16-packw.cc &

wait
