#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/f32-f16-vcvt/neonfp16.c.in -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/vcvt-neonfp16-x8.c &
tools/xngen src/f32-f16-vcvt/neonfp16.c.in -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/vcvt-neonfp16-x16.c &

################################# x86 256-bit #################################
tools/xngen src/f32-f16-vcvt/f16c.c.in -D BATCH_TILE=8  -o src/f32-f16-vcvt/gen/vcvt-f16c-x8.c &
tools/xngen src/f32-f16-vcvt/f16c.c.in -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/vcvt-f16c-x16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-f16-vcvt/avx512skx.c.in -D BATCH_TILE=16 -o src/f32-f16-vcvt/gen/vcvt-avx512skx-x16.c &
tools/xngen src/f32-f16-vcvt/avx512skx.c.in -D BATCH_TILE=32 -o src/f32-f16-vcvt/gen/vcvt-avx512skx-x32.c &

#################################### Scalar ###################################
tools/xngen src/f32-f16-vcvt/scalar-float.c.in -D BATCH_TILE=1 -o src/f32-f16-vcvt/gen/vcvt-scalar-float-x1.c &
tools/xngen src/f32-f16-vcvt/scalar-float.c.in -D BATCH_TILE=2 -o src/f32-f16-vcvt/gen/vcvt-scalar-float-x2.c &
tools/xngen src/f32-f16-vcvt/scalar-float.c.in -D BATCH_TILE=3 -o src/f32-f16-vcvt/gen/vcvt-scalar-float-x3.c &
tools/xngen src/f32-f16-vcvt/scalar-float.c.in -D BATCH_TILE=4 -o src/f32-f16-vcvt/gen/vcvt-scalar-float-x4.c &

################################## Unit tests #################################
tools/generate-vcvt-test.py --spec test/f32-f16-vcvt.yaml --output test/f32-f16-vcvt.cc &

wait
