#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QS8 -D WASM=0 -o src/qs8-f32-gemm/gen/qs8-f32-gemm-1x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QS8 -D WASM=0 -o src/qs8-f32-gemm/gen/qs8-f32-gemm-1x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=1 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QS8 -D WASM=0 -o src/qs8-f32-gemm/gen/qs8-f32-gemm-1x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=2 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QS8 -D WASM=0 -o src/qs8-f32-gemm/gen/qs8-f32-gemm-2x2-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QS8 -D WASM=0 -o src/qs8-f32-gemm/gen/qs8-f32-gemm-2x4-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=2 -D NR=8 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QS8 -D WASM=0 -o src/qs8-f32-gemm/gen/qs8-f32-gemm-2x8-minmax-scalar.c &
tools/xngen src/qs8-gemm/scalar.c.in -D MR=4 -D NR=4 -D REQUANTIZATION= -D VARIANT= -D DATATYPE=QS8 -D WASM=0 -o src/qs8-f32-gemm/gen/qs8-f32-gemm-4x4-minmax-scalar.c &

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/qs8-f32-gemm-minmax.yaml --output test/qs8-f32-gemm-minmax.cc

wait
