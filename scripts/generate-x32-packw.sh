#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x32-packw/scalar.c.in -D NR=2 -D KUNROLL=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x2-scalar-int.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=4 -D KUNROLL=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x4-scalar-int.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=2 -D KUNROLL=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x2-scalar-float.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=4 -D KUNROLL=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x4-scalar-float.c &

################################### ARM NEON ##################################
### NR multiple of 4
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8-neon.c &
tools/xngen src/x32-packw/neon.c.in -D NR=12 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x12-neon.c &

### NR2 micro-kernels
tools/xngen src/x32-packw/NR2-neon.c.in -D NR=2 -D KUNROLL=2 -o src/x32-packw/gen/x32-packw-x2-neon.c &

################################## Unit tests #################################
tools/generate-packw-test.py --spec test/x32-packw.yaml --output test/x32-packw.cc &

wait
