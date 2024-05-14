#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ARM NEON+FP16ARITH
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u32.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u32-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u32-acc4.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=40 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u40.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=40 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u40-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=40 -D ACCUMULATORS=5 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u40-acc5.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=48 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u48.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=48 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u48-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=48 -D ACCUMULATORS=3 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u48-acc3.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=64 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u64.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u64-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u64-acc4.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=72 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u72.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=72 -D ACCUMULATORS=3 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u72-acc3.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=80 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u80.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=80 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u80-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=80 -D ACCUMULATORS=5 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u80-acc5.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=96 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u96.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=96 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u96-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=96 -D ACCUMULATORS=3 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u96-acc3.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=96 -D ACCUMULATORS=6 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u96-acc6.c &

# x86 AVX2
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u16.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u16-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u32.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u32-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u32-acc4.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=40 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u40.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=40 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u40-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=40 -D ACCUMULATORS=5 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u40-acc5.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=48 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u48.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=48 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u48-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=48 -D ACCUMULATORS=3 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u48-acc3.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=64 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u64.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u64-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u64-acc4.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=72 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u72.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=72 -D ACCUMULATORS=3 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u72-acc3.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=80 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u80.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=80 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u80-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=80 -D ACCUMULATORS=5 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u80-acc5.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=96 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u96.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=96 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u96-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=96 -D ACCUMULATORS=3 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u96-acc3.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=96 -D ACCUMULATORS=6 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u96-acc6.c &

wait
