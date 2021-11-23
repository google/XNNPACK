#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/s8-ibilinear/neon.c.in -D CHANNEL_TILE=8  -D PIXEL_TILE=1 -D DATATYPE=S8 -o src/s8-ibilinear/gen/neon-c8.c &
tools/xngen src/s8-ibilinear/neon.c.in -D CHANNEL_TILE=16 -D PIXEL_TILE=1 -D DATATYPE=S8 -o src/s8-ibilinear/gen/neon-c16.c &

tools/xngen src/s8-ibilinear/neon.c.in -D CHANNEL_TILE=8  -D PIXEL_TILE=1 -D DATATYPE=U8 -o src/u8-ibilinear/gen/neon-c8.c &
tools/xngen src/s8-ibilinear/neon.c.in -D CHANNEL_TILE=16 -D PIXEL_TILE=1 -D DATATYPE=U8 -o src/u8-ibilinear/gen/neon-c16.c &

################################## Unit tests #################################
tools/generate-ibilinear-test.py --spec test/s8-ibilinear.yaml --output test/s8-ibilinear.cc &
tools/generate-ibilinear-test.py --spec test/u8-ibilinear.yaml --output test/u8-ibilinear.cc &

wait
