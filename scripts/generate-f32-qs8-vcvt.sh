#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neon-x8.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neon-x16.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neon-x24.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neon-x32.c &

tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neon-x8.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neon-x16.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neon-x24.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neon-x32.c &

tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neonv8-x8.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neonv8-x16.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neonv8-x24.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neonv8-x32.c &

tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neonv8-x8.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neonv8-x16.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neonv8-x24.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neonv8-x32.c &

################################# x86 128-bit #################################
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse2-x8.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse2-x16.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse2-x24.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse2-x32.c &

tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse41-x8.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse41-x16.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse41-x24.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse41-x32.c &

tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-sse2-x8.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-sse2-x16.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-sse2-x24.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-sse2-x32.c &

################################## Unit tests #################################
tools/generate-vcvt-test.py --spec test/f32-qs8-vcvt.yaml --output test/f32-qs8-vcvt.cc &
tools/generate-vcvt-test.py --spec test/f32-qu8-vcvt.yaml --output test/f32-qu8-vcvt.cc &

wait
