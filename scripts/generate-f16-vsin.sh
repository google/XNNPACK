#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f16-vsin/rational-3-2.c.in -D FUN=SIN -D ARCH=scalar        -D BATCH_TILES=1,2,4,8   -D DIV=DIV -o src/f16-vsin/gen/f16-vsin-scalar-rational-3-2-div.c &
tools/xngen src/f16-vsin/rational-3-2.c.in -D FUN=SIN -D ARCH=neonfp16arith -D BATCH_TILES=8,16,32   -D DIV=DIV -o src/f16-vsin/gen/f16-vsin-neonfp16arith-rational-3-2-div.c &
tools/xngen src/f16-vsin/rational-3-2.c.in -D FUN=SIN -D ARCH=avx512fp16    -D BATCH_TILES=32,64,96  -D DIV=DIV -o src/f16-vsin/gen/f16-vsin-avx512fp16-rational-3-2-div.c &

wait
