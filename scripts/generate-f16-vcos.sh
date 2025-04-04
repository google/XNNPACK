#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f16-vsin/rational-3-2.c.in -D FUN=COS -D ARCH=scalar -D BATCH_TILES=1,2,4,8 -D DIV=DIV -o src/f16-vcos/gen/f16-vcos-scalar-rational-3-2-div.c &
tools/xngen src/f16-vsin/rational-3-2.c.in -D FUN=COS -D ARCH=neonfp16arith -D BATCH_TILES=8,16,32 -D DIV=DIV -o src/f16-vcos/gen/f16-vcos-neonfp16arith-rational-3-2-div.c &
# Currently disabled since this causes compiler issues with clang-18
# (see https://github.com/llvm/llvm-project/issues/132013).
# tools/xngen src/f16-vsin/rational-3-2.c.in -D FUN=COS -D ARCH=avx512fp16 -D BATCH_TILES=32,64,96 -D DIV=DIV -o src/f16-vcos/gen/f16-vcos-avx512fp16-rational-3-2-div.c &

wait
