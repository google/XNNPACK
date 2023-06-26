#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Microkernels without unrolling
tools/xngen src/f32-spmm/scalar.c.in -D MR=1 -D NR=1 -D UNROLL=1 -D DATATYPE=QC8 -o src/f32-qc8w-spmm/gen/f32-qc8w-spmm-1x1-minmax-scalar.c &
tools/xngen src/f32-spmm/scalar.c.in -D MR=2 -D NR=1 -D UNROLL=1 -D DATATYPE=QC8 -o src/f32-qc8w-spmm/gen/f32-qc8w-spmm-2x1-minmax-scalar.c &
tools/xngen src/f32-spmm/scalar.c.in -D MR=4 -D NR=1 -D UNROLL=1 -D DATATYPE=QC8 -o src/f32-qc8w-spmm/gen/f32-qc8w-spmm-4x1-minmax-scalar.c &
tools/xngen src/f32-spmm/scalar.c.in -D MR=8 -D NR=1 -D UNROLL=1 -D DATATYPE=QC8 -o src/f32-qc8w-spmm/gen/f32-qc8w-spmm-8x1-minmax-scalar.c &
tools/xngen src/f32-spmm/scalar.c.in -D MR=8 -D NR=2 -D UNROLL=1 -D DATATYPE=QC8 -o src/f32-qc8w-spmm/gen/f32-qc8w-spmm-8x2-minmax-scalar.c &
tools/xngen src/f32-spmm/scalar.c.in -D MR=8 -D NR=4 -D UNROLL=1 -D DATATYPE=QC8 -o src/f32-qc8w-spmm/gen/f32-qc8w-spmm-8x4-minmax-scalar.c &

wait
