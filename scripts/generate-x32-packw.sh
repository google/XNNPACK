#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x32-packw/scalar.c.in -D NR=2 -D KR=4 -o src/x32-packw/gen/x32-packw-x2-scalar.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=4 -D KR=4 -o src/x32-packw/gen/x32-packw-x4-scalar.c &

################################## Unit tests #################################
tools/generate-packw-test.py --spec test/x32-packw.yaml --output test/x32-packw.cc &

wait
