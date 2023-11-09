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

#################################### Scalar ###################################
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=1 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-fmagic-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=2 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-fmagic-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=3 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-fmagic-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=4 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-fmagic-u4.c &

tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=1 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-imagic-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=2 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-imagic-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=3 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-imagic-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=4 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-imagic-u4.c &