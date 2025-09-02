#!/bin/sh
# Copyright 2024-2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### NEON #####################################
tools/xngen src/f16-f32acc-rdsum2/neon.c.in -D CHANNELS_BATCHES=16,32,64 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum2/gen/f16-f32acc-rdsum2-7p7x-minmax-neonfp16arith.c &

################################## x86 AVX ####################################
tools/xngen src/f16-f32acc-rdsum2/avx.c.in -D CHANNELS_BATCHES=16,32,64,128 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum2/gen/f16-f32acc-rdsum2-7p7x-f16c.c &

################################## x86 AVX512 #################################
tools/xngen src/f16-f32acc-rdsum2/avx512skx.c.in -D CHANNELS_BATCHES=16,32,64,128 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum2/gen/f16-f32acc-rdsum2-7p7x-avx512skx.c &

wait
