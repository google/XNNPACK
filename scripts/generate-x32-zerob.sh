#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################

### Generic C micro-kernels

tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=2 -D CHANNEL_SUBTILE=1 -D TYPE=uint32_t -o src/x32-zerob/gen/x32-zerob-2c1s1r-scalar-int.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=2 -D CHANNEL_SUBTILE=2 -D TYPE=uint32_t -o src/x32-zerob/gen/x32-zerob-2c2s1r-scalar-int.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=4 -D CHANNEL_SUBTILE=1 -D TYPE=uint32_t -o src/x32-zerob/gen/x32-zerob-4c1s1r-scalar-int.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=4 -D CHANNEL_SUBTILE=4 -D TYPE=uint32_t -o src/x32-zerob/gen/x32-zerob-4c4s1r-scalar-int.c &

tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=2 -D CHANNEL_SUBTILE=1 -D TYPE=float    -o src/x32-zerob/gen/x32-zerob-2c1s1r-scalar-float.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=2 -D CHANNEL_SUBTILE=2 -D TYPE=float    -o src/x32-zerob/gen/x32-zerob-2c2s1r-scalar-float.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=4 -D CHANNEL_SUBTILE=1 -D TYPE=float    -o src/x32-zerob/gen/x32-zerob-4c1s1r-scalar-float.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=4 -D CHANNEL_SUBTILE=4 -D TYPE=float    -o src/x32-zerob/gen/x32-zerob-4c4s1r-scalar-float.c &

################################### x86 SSE ###################################

tools/xngen src/x32-packb/sse2.c.in -D BIAS=0 -D CHANNEL_TILE=4  -D CHANNEL_SUBTILE=4  -o src/x32-zerob/gen/x32-zerob-4c4s4r-sse2.c &
tools/xngen src/x32-packb/sse2.c.in -D BIAS=0 -D CHANNEL_TILE=8  -D CHANNEL_SUBTILE=4  -o src/x32-zerob/gen/x32-zerob-8c4s4r-sse2.c &
tools/xngen src/x32-packb/sse2.c.in -D BIAS=0 -D CHANNEL_TILE=8  -D CHANNEL_SUBTILE=8  -o src/x32-zerob/gen/x32-zerob-8c8s4r-sse2.c &
tools/xngen src/x32-packb/sse2.c.in -D BIAS=0 -D CHANNEL_TILE=16 -D CHANNEL_SUBTILE=4  -o src/x32-zerob/gen/x32-zerob-16c4s4r-sse2.c &
tools/xngen src/x32-packb/sse2.c.in -D BIAS=0 -D CHANNEL_TILE=16 -D CHANNEL_SUBTILE=16 -o src/x32-zerob/gen/x32-zerob-16c16s4r-sse2.c &

################################## Unit tests #################################
tools/generate-packb-test.py --spec test/x32-zerob.yaml --output test/x32-zerob.cc &

wait
