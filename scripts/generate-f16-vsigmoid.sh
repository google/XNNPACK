#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=8  -D DIV_ALGO=div   -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-div-x8.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=16 -D DIV_ALGO=div   -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-div-x16.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=24 -D DIV_ALGO=div   -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-div-x24.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=32 -D DIV_ALGO=div   -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-div-x32.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=40 -D DIV_ALGO=div   -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-div-x40.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=48 -D DIV_ALGO=div   -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-div-x48.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=56 -D DIV_ALGO=div   -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-div-x56.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=64 -D DIV_ALGO=div   -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-div-x64.c &

tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=8  -D DIV_ALGO=recpe -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-recpe-x8.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=16 -D DIV_ALGO=recpe -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-recpe-x16.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=24 -D DIV_ALGO=recpe -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-recpe-x24.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=32 -D DIV_ALGO=recpe -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-recpe-x32.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=40 -D DIV_ALGO=recpe -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-recpe-x40.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=48 -D DIV_ALGO=recpe -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-recpe-x48.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=56 -D DIV_ALGO=recpe -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-recpe-x56.c &
tools/xngen src/f16-vsigmoid/neonfp16arith.c.in -D BATCH_TILE=64 -D DIV_ALGO=recpe -o src/f16-vsigmoid/gen/vsigmoid-neonfp16arith-rr1-p3-recpe-x64.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f16-vsigmoid.yaml --output test/f16-vsigmoid.cc &

wait
