#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################# ARM NEONFP16 ################################
tools/xngen src/f16-f32acc-rsum/neonfp16arith.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16arith-u4.c &
tools/xngen src/f16-f32acc-rsum/neonfp16arith.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16arith-u8.c &
tools/xngen src/f16-f32acc-rsum/neonfp16arith.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16arith-u16-acc2.c &
tools/xngen src/f16-f32acc-rsum/neonfp16arith.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16arith-u24-acc3.c &
tools/xngen src/f16-f32acc-rsum/neonfp16arith.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16arith-u32-acc2.c &
tools/xngen src/f16-f32acc-rsum/neonfp16arith.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-neonfp16arith-u32-acc4.c &

################################### x86 F16C ##################################
tools/xngen src/f16-f32acc-rsum/f16c.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-f16c-u8.c &
tools/xngen src/f16-f32acc-rsum/f16c.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-f16c-u16-acc2.c &
tools/xngen src/f16-f32acc-rsum/f16c.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-f16c-u24-acc3.c &
tools/xngen src/f16-f32acc-rsum/f16c.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-f16c-u32-acc2.c &
tools/xngen src/f16-f32acc-rsum/f16c.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-f16c-u32-acc4.c &

################################## x86 AVX512 #################################
tools/xngen src/f16-f32acc-rsum/avx512skx.c.in -D BATCH_TILE=16  -D ACCUMULATORS=1 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u16.c &
tools/xngen src/f16-f32acc-rsum/avx512skx.c.in -D BATCH_TILE=32  -D ACCUMULATORS=2 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u32-acc2.c &
tools/xngen src/f16-f32acc-rsum/avx512skx.c.in -D BATCH_TILE=48  -D ACCUMULATORS=3 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u48-acc3.c &
tools/xngen src/f16-f32acc-rsum/avx512skx.c.in -D BATCH_TILE=64  -D ACCUMULATORS=2 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u64-acc2.c &
tools/xngen src/f16-f32acc-rsum/avx512skx.c.in -D BATCH_TILE=64  -D ACCUMULATORS=4 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u64-acc4.c &
tools/xngen src/f16-f32acc-rsum/avx512skx.c.in -D BATCH_TILE=128 -D ACCUMULATORS=4 -o src/f16-f32acc-rsum/gen/f16-f32acc-rsum-avx512skx-u128-acc4.c &

wait
