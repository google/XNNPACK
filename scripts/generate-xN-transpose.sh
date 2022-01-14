#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=int8_t SIZE=8 -o src/x8-transpose/gen/1x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=4 TYPE=int8_t SIZE=8 -o src/x8-transpose/gen/1x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=int8_t SIZE=8 -o src/x8-transpose/gen/2x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=int8_t SIZE=8 -o src/x8-transpose/gen/2x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=4 TYPE=int8_t SIZE=8 -o src/x8-transpose/gen/2x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=int8_t SIZE=8 -o src/x8-transpose/gen/4x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=int8_t SIZE=8 -o src/x8-transpose/gen/4x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=4 TYPE=int8_t SIZE=8 -o src/x8-transpose/gen/4x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/1x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=4 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/1x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/2x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/2x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=4 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/2x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/4x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/4x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=4 TYPE=int16_t SIZE=16 -o src/x16-transpose/gen/4x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=int SIZE=32 -o src/x32-transpose/gen/1x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=4 TYPE=int SIZE=32 -o src/x32-transpose/gen/1x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=int SIZE=32 -o src/x32-transpose/gen/2x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=int SIZE=32 -o src/x32-transpose/gen/2x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=4 TYPE=int SIZE=32 -o src/x32-transpose/gen/2x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=int SIZE=32 -o src/x32-transpose/gen/4x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=int SIZE=32 -o src/x32-transpose/gen/4x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=4 TYPE=int SIZE=32 -o src/x32-transpose/gen/4x4-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=float SIZE=32 -o src/x32-transpose/gen/1x2-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=4 TYPE=float SIZE=32 -o src/x32-transpose/gen/1x4-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=float SIZE=32 -o src/x32-transpose/gen/2x1-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=float SIZE=32 -o src/x32-transpose/gen/2x2-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=4 TYPE=float SIZE=32 -o src/x32-transpose/gen/2x4-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=float SIZE=32 -o src/x32-transpose/gen/4x1-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=float SIZE=32 -o src/x32-transpose/gen/4x2-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=4 TYPE=float SIZE=32 -o src/x32-transpose/gen/4x4-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=int64_t SIZE=64 -o src/x64-transpose/gen/1x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=int64_t SIZE=64 -o src/x64-transpose/gen/2x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=int64_t SIZE=64 -o src/x64-transpose/gen/2x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=int64_t SIZE=64 -o src/x64-transpose/gen/4x1-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=int64_t SIZE=64 -o src/x64-transpose/gen/4x2-scalar-int.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=1 TILE_WIDTH=2 TYPE=double SIZE=64 -o src/x64-transpose/gen/1x2-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=1 TYPE=double SIZE=64 -o src/x64-transpose/gen/2x1-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 TYPE=double SIZE=64 -o src/x64-transpose/gen/2x2-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=1 TYPE=double SIZE=64 -o src/x64-transpose/gen/4x1-scalar-float.c &
tools/xngen src/x32-transpose/scalar.c.in -D TILE_HEIGHT=4 TILE_WIDTH=2 TYPE=double SIZE=64 -o src/x64-transpose/gen/4x2-scalar-float.c &

################################## Unit tests #################################
tools/generate-transpose-test.py --spec test/x8-transpose.yaml --output=test/x8-transpose.cc &
tools/generate-transpose-test.py --spec test/x16-transpose.yaml --output=test/x16-transpose.cc &
tools/generate-transpose-test.py --spec test/x32-transpose.yaml --output=test/x32-transpose.cc &
tools/generate-transpose-test.py --spec test/x64-transpose.yaml --output=test/x64-transpose.cc &

wait
