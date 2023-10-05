#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### x86 AVX2 ##################################
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=8  -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u8.c &
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=16 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u16.c &
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=24 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u24.c &
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=32 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u32.c &
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=40 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u40.c &
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=48 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u48.c &
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=56 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u56.c &
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=64 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u64.c &
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=72 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u72.c &
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=80 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u80.c &
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=88 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u88.c &
tools/xngen src/f32-vscaleextexp/avx2-p5.c.in -D BATCH_TILE=96 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx2-p5-u96.c &

################################# x86 AVX512F #################################
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=16  -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u16.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=32  -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u32.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=48  -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u48.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=64  -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u64.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=80  -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u80.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=96  -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u96.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=112 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u112.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=128 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u128.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=144 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u144.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=160 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u160.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=176 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u176.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=192 -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u192.c &

wait
