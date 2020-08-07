#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### C4 micro-kernels
tools/xngen src/qu8-gemm/MRxNRc4-minmax-scalar.c.in -D MR=8  -D NR=8 -o src/qu8-gemm/gen/8x8c4-minmax-scalar.c
tools/xngen src/qu8-gemm/MRxNRc4-minmax-scalar.c.in -D MR=12 -D NR=4 -o src/qu8-gemm/gen/12x4c4-minmax-scalar.c
