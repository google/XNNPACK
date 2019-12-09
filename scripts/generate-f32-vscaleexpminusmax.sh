#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### x86 AVX2 ##################################
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=8 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x8.c
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=16 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x16.c
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=24 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x24.c
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=32 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x32.c
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=40 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x40.c
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=48 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x48.c
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=56 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x56.c
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=64 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x64.c
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=72 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x72.c
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=80 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x80.c
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=88 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x88.c
tools/xngen src/f32-vscaleexpminusmax/avx2-p5.c.in -D ELEMENTS_TILE=96 -o src/f32-vscaleexpminusmax/gen/avx2-p5-x96.c

################################# x86 AVX512F #################################
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=16 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x16.c
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=32 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x32.c
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=48 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x48.c
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=64 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x64.c
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=80 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x80.c
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=96 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x96.c
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=112 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x112.c
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=128 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x128.c
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=144 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x144.c
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=160 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x160.c
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=176 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x176.c
tools/xngen src/f32-vscaleexpminusmax/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=192 -o src/f32-vscaleexpminusmax/gen/avx512f-p5-scalef-x192.c

################################## Unit tests #################################
tools/generate-vscaleexpminusmax-test.py --spec test/f32-vscaleexpminusmax.yaml --output test/f32-vscaleexpminusmax.cc
