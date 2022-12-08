#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=8  -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-x8.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=16 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-x16.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=24 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-x24.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=32 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-x32.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=40 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-x40.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=48 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-x48.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=56 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-x56.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=64 -D DIV_ALGO=DIV -o src/f16-vsigmoid/gen/f16-vsigmoid-aarch64-neonfp16arith-rr2-p2-div-x64.c &

tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=8  -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-x8.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=16 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-x16.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=24 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-x24.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=32 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-x32.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=40 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-x40.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=48 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-x48.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=56 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-x56.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=64 -D DIV_ALGO=NR1FMA -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1fma-x64.c &

tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=8  -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-x8.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=16 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-x16.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=24 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-x24.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=32 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-x32.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=40 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-x40.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=48 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-x48.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=56 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-x56.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=64 -D DIV_ALGO=NR1RECPS -o src/f16-vsigmoid/gen/f16-vsigmoid-neonfp16arith-rr2-p2-nr1recps-x64.c &

################################### x86 AVX2 ##################################
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=8  -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-x8.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=16 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-x16.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=24 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-x24.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=32 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-x32.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=40 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-x40.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=48 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-x48.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=56 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-x56.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=64 -D DIV_ALGO=div -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-div-x64.c &

tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=8  -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-x8.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=16 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-x16.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=24 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-x24.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=32 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-x32.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=40 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-x40.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=48 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-x48.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=56 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-x56.c &
tools/xngen src/f16-vsigmoid/avx2.c.in -D BATCH_TILE=64 -D DIV_ALGO=rcp -o src/f16-vsigmoid/gen/f16-vsigmoid-avx2-rr1-p2-rcp-x64.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f16-vsigmoid.yaml --output test/f16-vsigmoid.cc &

wait
