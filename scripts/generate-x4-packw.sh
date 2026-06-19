#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### C8 packing
tools/xngen src/x4-packw/scalar.c.in -D NR=8  -D KR=8 -D SR=1 -o src/qs8-qc4w-packw/gen/qs8-qc4w-packw-x8c8-gemm-goi-scalar.c &
tools/xngen src/x4-packw/scalar.c.in -D NR=16 -D KR=8 -D SR=1 -o src/qs8-qc4w-packw/gen/qs8-qc4w-packw-x16c8-gemm-goi-scalar.c &
tools/xngen src/x4-packw/scalar.c.in -D NR=32 -D KR=8 -D SR=1 -o src/qs8-qc4w-packw/gen/qs8-qc4w-packw-x32c8-gemm-goi-scalar.c &

wait
