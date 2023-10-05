#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-velu/neonfp16arith-rr1-p3.c.in -D BATCH_TILE=8  -o src/f16-velu/gen/f16-velu-neonfp16arith-rr1-p3-u8.c &
tools/xngen src/f16-velu/neonfp16arith-rr1-p3.c.in -D BATCH_TILE=16 -o src/f16-velu/gen/f16-velu-neonfp16arith-rr1-p3-u16.c &

################################### x86 AVX2 ##################################
tools/xngen src/f16-velu/avx2-rr1-p3.c.in -D BATCH_TILE=8  -o src/f16-velu/gen/f16-velu-avx2-rr1-p3-u8.c &
tools/xngen src/f16-velu/avx2-rr1-p3.c.in -D BATCH_TILE=16 -o src/f16-velu/gen/f16-velu-avx2-rr1-p3-u16.c &

wait
