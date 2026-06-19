#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

############################## ARM NEON+FP16ARITH #############################
tools/xngen src/f16-rsum/neonfp16arith.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -o src/f16-rsum/gen/f16-rsum-neonfp16arith-u8.c &
tools/xngen src/f16-rsum/neonfp16arith.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f16-rsum/gen/f16-rsum-neonfp16arith-u16-acc2.c &
tools/xngen src/f16-rsum/neonfp16arith.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -o src/f16-rsum/gen/f16-rsum-neonfp16arith-u24-acc3.c &
tools/xngen src/f16-rsum/neonfp16arith.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f16-rsum/gen/f16-rsum-neonfp16arith-u32-acc2.c &
tools/xngen src/f16-rsum/neonfp16arith.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f16-rsum/gen/f16-rsum-neonfp16arith-u32-acc4.c &

############################## x86 AVX512FP16 #############################
tools/xngen src/f16-rsum/avx512fp16.c.in -D BATCH_TILE=32  -D ACCUMULATORS=1 -o src/f16-rsum/gen/f16-rsum-avx512fp16-u32.c &
tools/xngen src/f16-rsum/avx512fp16.c.in -D BATCH_TILE=64  -D ACCUMULATORS=2 -o src/f16-rsum/gen/f16-rsum-avx512fp16-u64-acc2.c &
tools/xngen src/f16-rsum/avx512fp16.c.in -D BATCH_TILE=96  -D ACCUMULATORS=3 -o src/f16-rsum/gen/f16-rsum-avx512fp16-u96-acc3.c &
tools/xngen src/f16-rsum/avx512fp16.c.in -D BATCH_TILE=128 -D ACCUMULATORS=2 -o src/f16-rsum/gen/f16-rsum-avx512fp16-u128-acc2.c &
tools/xngen src/f16-rsum/avx512fp16.c.in -D BATCH_TILE=128 -D ACCUMULATORS=4 -o src/f16-rsum/gen/f16-rsum-avx512fp16-u128-acc4.c &

wait
