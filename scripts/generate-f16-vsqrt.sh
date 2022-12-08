#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vsqrt/neonfp16arith-sqrt.c.in -D BATCH_TILE=8  -o src/f16-vsqrt/gen/f16-vsqrt-aarch64-neonfp16arith-sqrt-x8.c &
tools/xngen src/f16-vsqrt/neonfp16arith-sqrt.c.in -D BATCH_TILE=16 -o src/f16-vsqrt/gen/f16-vsqrt-aarch64-neonfp16arith-sqrt-x16.c &

tools/xngen src/f16-vsqrt/neonfp16arith-nr1fma1adj.c.in -D BATCH_TILE=8  -o src/f16-vsqrt/gen/f16-vsqrt-neonfp16arith-nr1fma1adj-x8.c &
tools/xngen src/f16-vsqrt/neonfp16arith-nr1fma1adj.c.in -D BATCH_TILE=16 -o src/f16-vsqrt/gen/f16-vsqrt-neonfp16arith-nr1fma1adj-x16.c &
tools/xngen src/f16-vsqrt/neonfp16arith-nr1fma1adj.c.in -D BATCH_TILE=24 -o src/f16-vsqrt/gen/f16-vsqrt-neonfp16arith-nr1fma1adj-x24.c &
tools/xngen src/f16-vsqrt/neonfp16arith-nr1fma1adj.c.in -D BATCH_TILE=32 -o src/f16-vsqrt/gen/f16-vsqrt-neonfp16arith-nr1fma1adj-x32.c &

################################### ARM FP16 ##################################
tools/xngen src/f16-vsqrt/fp16arith-sqrt.c.in -D BATCH_TILE=1 -o src/f16-vsqrt/gen/f16-vsqrt-fp16arith-sqrt-x1.c &
tools/xngen src/f16-vsqrt/fp16arith-sqrt.c.in -D BATCH_TILE=2 -o src/f16-vsqrt/gen/f16-vsqrt-fp16arith-sqrt-x2.c &
tools/xngen src/f16-vsqrt/fp16arith-sqrt.c.in -D BATCH_TILE=4 -o src/f16-vsqrt/gen/f16-vsqrt-fp16arith-sqrt-x4.c &

################################### x86 F16C ##################################
tools/xngen src/f16-vsqrt/f16c-sqrt.c.in -D BATCH_TILE=8  -o src/f16-vsqrt/gen/f16-vsqrt-f16c-sqrt-x8.c &
tools/xngen src/f16-vsqrt/f16c-sqrt.c.in -D BATCH_TILE=16 -o src/f16-vsqrt/gen/f16-vsqrt-f16c-sqrt-x16.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f16-vsqrt.yaml --output test/f16-vsqrt.cc &

wait
