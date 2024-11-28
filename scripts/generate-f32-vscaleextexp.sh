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

################################# x86 AVX512F #################################
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=16  -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u16.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=32  -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u32.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=48  -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u48.c &
tools/xngen src/f32-vscaleextexp/avx512f-p5-scalef.c.in -D BATCH_TILE=64  -o src/f32-vscaleextexp/gen/f32-vscaleextexp-avx512f-p5-scalef-u64.c &

wait
