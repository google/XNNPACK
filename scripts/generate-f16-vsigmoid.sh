#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=8  -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-u8.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=16 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-u16.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=24 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-u24.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=32 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-u32.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=40 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-u40.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=48 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-u48.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=56 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-u56.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=64 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-u64.c &

tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=8  -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-u8.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=16 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-u16.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=24 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-u24.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=32 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-u32.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=40 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-u40.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=48 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-u48.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=56 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-u56.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=64 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-u64.c &

tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=8  -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-u8.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=16 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-u16.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=24 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-u24.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=32 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-u32.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=40 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-u40.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=48 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-u48.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=56 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-u56.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=64 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-u64.c &

################################### x86 AVX2 ##################################
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=8  -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u8.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=16 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u16.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=24 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u24.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=32 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u32.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=40 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u40.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=48 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u48.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=56 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u56.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=64 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-u64.c &

tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=8  -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u8.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=16 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u16.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=24 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u24.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=32 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u32.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=40 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u40.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=48 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u48.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=56 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u56.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=64 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-u64.c &

wait
