#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### x86 AVX2 ##################################
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=64 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u64.c &
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u64-acc2.c &
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u64-acc4.c &
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=72 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u72.c &
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=72 -D ACCUMULATORS=3 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u72-acc3.c &
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=80 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u80.c &
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=80 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u80-acc2.c &
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=80 -D ACCUMULATORS=5 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u80-acc5.c &
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=96 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u96.c &
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=96 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u96-acc2.c &
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=96 -D ACCUMULATORS=3 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u96-acc3.c &
tools/xngen src/f32-raddextexp/avx2-p5.c.in -D BATCH_TILE=96 -D ACCUMULATORS=6 -o src/f32-raddextexp/gen/f32-raddextexp-avx2-p5-u96-acc6.c &

################################# x86 AVX512F #################################
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=128 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u128.c &
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=128 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u128-acc2.c &
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=128 -D ACCUMULATORS=4 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u128-acc4.c &
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=144 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u144.c &
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=144 -D ACCUMULATORS=3 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u144-acc3.c &
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=160 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u160.c &
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=160 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u160-acc2.c &
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=160 -D ACCUMULATORS=5 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u160-acc5.c &
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=192 -D ACCUMULATORS=1 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u192.c &
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=192 -D ACCUMULATORS=2 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u192-acc2.c &
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=192 -D ACCUMULATORS=3 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u192-acc3.c &
tools/xngen src/f32-raddextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=192 -D ACCUMULATORS=6 -o src/f32-raddextexp/gen/f32-raddextexp-avx512f-p5-scalef-u192-acc6.c &

wait
