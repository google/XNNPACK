#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### SSE2 ###################################
tools/xngen src/x32-transpose/sse2.c.in -D TILE_HEIGHT=16 TILE_WIDTH=16 SIZE=8 -o src/x8-transpose/gen/16x16-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D TILE_HEIGHT=8 TILE_WIDTH=8 SIZE=16 -o src/x16-transpose/gen/8x8-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D TILE_HEIGHT=4 TILE_WIDTH=4 SIZE=32 -o src/x32-transpose/gen/4x4-sse2.c &
tools/xngen src/x32-transpose/sse2.c.in -D TILE_HEIGHT=2 TILE_WIDTH=2 SIZE=64 -o src/x64-transpose/gen/2x2-sse2.c &

################################## Unit tests #################################
tools/generate-transpose-test.py --spec test/x8-transpose.yaml --output=test/x8-transpose.cc &
tools/generate-transpose-test.py --spec test/x16-transpose.yaml --output=test/x16-transpose.cc &
tools/generate-transpose-test.py --spec test/x32-transpose.yaml --output=test/x32-transpose.cc &
tools/generate-transpose-test.py --spec test/x64-transpose.yaml --output=test/x64-transpose.cc &

wait
