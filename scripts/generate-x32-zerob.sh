#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=2 -D CHANNEL_SUBTILE=1 -D TYPE=uint32_t -o src/x32-zerob/gen/x32-zerob-2c1s1r-gemm-scalar-int.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=2 -D CHANNEL_SUBTILE=2 -D TYPE=uint32_t -o src/x32-zerob/gen/x32-zerob-2c2s1r-gemm-scalar-int.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=4 -D CHANNEL_SUBTILE=1 -D TYPE=uint32_t -o src/x32-zerob/gen/x32-zerob-4c1s1r-gemm-scalar-int.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=4 -D CHANNEL_SUBTILE=4 -D TYPE=uint32_t -o src/x32-zerob/gen/x32-zerob-4c4s1r-gemm-scalar-int.c &

tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=2 -D CHANNEL_SUBTILE=1 -D TYPE=float    -o src/x32-zerob/gen/x32-zerob-2c1s1r-gemm-scalar-float.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=2 -D CHANNEL_SUBTILE=2 -D TYPE=float    -o src/x32-zerob/gen/x32-zerob-2c2s1r-gemm-scalar-float.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=4 -D CHANNEL_SUBTILE=1 -D TYPE=float    -o src/x32-zerob/gen/x32-zerob-4c1s1r-gemm-scalar-float.c &
tools/xngen src/x32-packb/scalar.c.in -D BIAS=0 -D CHANNEL_TILE=4 -D CHANNEL_SUBTILE=4 -D TYPE=float    -o src/x32-zerob/gen/x32-zerob-4c4s1r-gemm-scalar-float.c &

wait
