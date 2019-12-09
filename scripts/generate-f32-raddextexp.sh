#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### x86 AVX2 ##################################
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=64 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/avx2-p5-x64.c
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=64 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/avx2-p5-x64-acc2.c
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=64 -D ACCUMULATORS=4 -o src/f32-raddextexp/gen/avx2-p5-x64-acc4.c
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=72 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/avx2-p5-x72.c
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=72 -D ACCUMULATORS=3 -o src/f32-raddextexp/gen/avx2-p5-x72-acc3.c
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=80 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/avx2-p5-x80.c
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=80 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/avx2-p5-x80-acc2.c
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=80 -D ACCUMULATORS=5 -o src/f32-raddextexp/gen/avx2-p5-x80-acc5.c
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=96 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/avx2-p5-x96.c
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=96 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/avx2-p5-x96-acc2.c
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=96 -D ACCUMULATORS=3 -o src/f32-raddextexp/gen/avx2-p5-x96-acc3.c
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D ELEMENTS_TILE=96 -D ACCUMULATORS=6 -o src/f32-raddextexp/gen/avx2-p5-x96-acc6.c

################################# x86 AVX512F #################################
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=128 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x128.c
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=128 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x128-acc2.c
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=128 -D ACCUMULATORS=4 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x128-acc4.c
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=144 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x144.c
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=144 -D ACCUMULATORS=3 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x144-acc3.c
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=160 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x160.c
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=160 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x160-acc2.c
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=160 -D ACCUMULATORS=5 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x160-acc5.c
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=192 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x192.c
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=192 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x192-acc2.c
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=192 -D ACCUMULATORS=3 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x192-acc3.c
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D ELEMENTS_TILE=192 -D ACCUMULATORS=6 -o src/f32-raddextexp/gen/avx512f-p5-scalef-x192-acc6.c

################################## Unit tests #################################
tools/generate-raddextexp-test.py --spec test/f32-raddextexp.yaml --output test/f32-raddextexp.cc
