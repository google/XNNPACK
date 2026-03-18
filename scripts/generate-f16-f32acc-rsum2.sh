#!/bin/sh
# Copyright 2023-2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################# ARM NEONFP16 ################################
tools/xngen src/f16-f32acc-rsum2/neonfp16arith.c.in -D BATCH_TILES=8,16,24,32 -o src/f16-f32acc-rsum2/gen/f16-f32acc-rsum2-neonfp16arith.c &

################################### x86 F16C ##################################
tools/xngen src/f16-f32acc-rsum2/f16c.c.in -D BATCH_TILES=8,16,24,32 -o src/f16-f32acc-rsum2/gen/f16-f32acc-rsum2-f16c.c &

################################## x86 AVX512 #################################
tools/xngen src/f16-f32acc-rsum2/avx512skx.c.in -D BATCH_TILES=16,32,48,64,128 -o src/f16-f32acc-rsum2/gen/f16-f32acc-rsum2-avx512skx.c &

wait
