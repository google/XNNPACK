#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################# x86 256-bit #################################
tools/xngen src/f16-f32-vcvt/f16c.c.in -D BATCH_TILE=8  -o src/f16-f32-vcvt/gen/vcvt-f16c-x8.c &
tools/xngen src/f16-f32-vcvt/f16c.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-f16c-x16.c &

################################# x86 512-bit #################################
tools/xngen src/f16-f32-vcvt/avx512skx.c.in -D BATCH_TILE=16 -o src/f16-f32-vcvt/gen/vcvt-avx512skx-x16.c &
tools/xngen src/f16-f32-vcvt/avx512skx.c.in -D BATCH_TILE=32 -o src/f16-f32-vcvt/gen/vcvt-avx512skx-x32.c &

################################## Unit tests #################################
tools/generate-vcvt-test.py --spec test/f16-f32-vcvt.yaml --output test/f16-f32-vcvt.cc &

wait
