#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-bf16-vcvt/scalar.c.in -D BATCH_TILE=1 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-scalar-u1.c &
tools/xngen src/f32-bf16-vcvt/scalar.c.in -D BATCH_TILE=2 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-scalar-u2.c &
tools/xngen src/f32-bf16-vcvt/scalar.c.in -D BATCH_TILE=3 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-scalar-u3.c &
tools/xngen src/f32-bf16-vcvt/scalar.c.in -D BATCH_TILE=4 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-scalar-u4.c &

wait
