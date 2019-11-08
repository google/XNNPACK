#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-bilinear/scalar.c.in -D CHANNEL_TILE=1 -D PIXEL_TILE=1 -o src/f32-bilinear/scalar-c1.c
tools/xngen src/f32-bilinear/scalar.c.in -D CHANNEL_TILE=2 -D PIXEL_TILE=1 -o src/f32-bilinear/scalar-c2.c
tools/xngen src/f32-bilinear/scalar.c.in -D CHANNEL_TILE=4 -D PIXEL_TILE=1 -o src/f32-bilinear/scalar-c4.c

################################### ARM NEON ##################################
tools/xngen src/f32-bilinear/neon.c.in -D CHANNEL_TILE=4 -D PIXEL_TILE=1 -D FMA=0 -o src/f32-bilinear/neon-c4.c
tools/xngen src/f32-bilinear/neon.c.in -D CHANNEL_TILE=8 -D PIXEL_TILE=1 -D FMA=0 -o src/f32-bilinear/neon-c8.c

tools/xngen src/f32-bilinear/neon.c.in -D CHANNEL_TILE=4 -D PIXEL_TILE=1 -D FMA=1 -o src/f32-bilinear/neonfma-c4.c
tools/xngen src/f32-bilinear/neon.c.in -D CHANNEL_TILE=8 -D PIXEL_TILE=1 -D FMA=1 -o src/f32-bilinear/neonfma-c8.c

#################################### PSIMD ####################################
tools/xngen src/f32-bilinear/psimd.c.in -D CHANNEL_TILE=4 -D PIXEL_TILE=1 -o src/f32-bilinear/psimd-c4.c
tools/xngen src/f32-bilinear/psimd.c.in -D CHANNEL_TILE=8 -D PIXEL_TILE=1 -o src/f32-bilinear/psimd-c8.c

################################### x86 SSE ###################################
tools/xngen src/f32-bilinear/sse.c.in -D CHANNEL_TILE=4 -D PIXEL_TILE=1 -o src/f32-bilinear/sse-c4.c
tools/xngen src/f32-bilinear/sse.c.in -D CHANNEL_TILE=8 -D PIXEL_TILE=1 -o src/f32-bilinear/sse-c8.c

################################## Unit tests #################################
tools/generate-bilinear-test.py --spec test/f32-bilinear.yaml --output test/f32-bilinear.cc
