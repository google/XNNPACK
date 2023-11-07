#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEONFP16ARITH ##################################
tools/xngen src/f16-qs8-vcvt/neonfp16arith.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-neonfp16arith-u8.c &
tools/xngen src/f16-qs8-vcvt/neonfp16arith.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-neonfp16arith-u16.c &
tools/xngen src/f16-qs8-vcvt/neonfp16arith.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-neonfp16arith-u24.c &
tools/xngen src/f16-qs8-vcvt/neonfp16arith.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-neonfp16arith-u32.c &
tools/xngen src/f16-qs8-vcvt/neonfp16arith.c.in -D BATCH_TILE=64 -D DATATYPE=QS8 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-neonfp16arith-u64.c &

wait
