#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
# C8 Packing
tools/xngen src/qb4-packw/kr-scalar.c.in -D NR=16 -D KR=8 -D -o src/qb4-packw/gen/qb4-packw-x16c8-gemm-goi-scalar.c

# C4 Packing
tools/xngen src/qb4-packw/kr-scalar.c.in -D NR=16 -D KR=4 -D -o src/qb4-packw/gen/qb4-packw-x16c4-gemm-goi-scalar.c

#################################### NeonDot ###################################
# C8 Packing
tools/xngen src/qb4-packw/c8-aarch64-neondot.c.in -D NR=16 -D KR=8 -D -o src/qb4-packw/gen/qb4-packw-x16c8-gemm-goi-aarch64-neondot.c

# C4 Packing
tools/xngen src/qb4-packw/c4-aarch64-neondot.c.in -D NR=16 -D KR=4 -D -o src/qb4-packw/gen/qb4-packw-x16c4-gemm-goi-aarch64-neondot.c
