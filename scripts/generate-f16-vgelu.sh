#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f16-vgelu/rational-6-4.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8 -o src/f16-vgelu/gen/f16-vgelu-scalar-rational-6-4-div.c &
tools/xngen src/f16-vgelu/rational-6-4.c.in -D ARCH=neonfp16arith -D BATCH_TILES=8,16,32 -o src/f16-vgelu/gen/f16-vgelu-neonfp16arith-rational-6-4-div.c &
tools/xngen src/f16-vgelu/rational-6-4.c.in -D ARCH=avx512fp16 -D BATCH_TILES=32,64,96 -o src/f16-vgelu/gen/f16-vgelu-avx512fp16-rational-6-4-div.c &

wait

################################## RISC-V RVV #################################
tools/xngen src/f16-vgelu/rvvfp16arith-rational-6-4.c.in -D LMUL=1 -o src/f16-vgelu/gen/f16-vgelu-rvvfp16arith-rational-6-4-div-u1v.c &
tools/xngen src/f16-vgelu/rvvfp16arith-rational-6-4.c.in -D LMUL=2 -o src/f16-vgelu/gen/f16-vgelu-rvvfp16arith-rational-6-4-div-u2v.c &
tools/xngen src/f16-vgelu/rvvfp16arith-rational-6-4.c.in -D LMUL=4 -o src/f16-vgelu/gen/f16-vgelu-rvvfp16arith-rational-6-4-div-u4v.c &
