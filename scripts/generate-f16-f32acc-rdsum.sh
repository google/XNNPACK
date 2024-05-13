#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### NEON #####################################
tools/xngen src/f16-f32acc-rdsum/neon.c.in -D CHANNELS_BATCH=16 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-minmax-neonfp16arith-c16.c &
tools/xngen src/f16-f32acc-rdsum/neon.c.in -D CHANNELS_BATCH=32 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-minmax-neonfp16arith-c32.c &
tools/xngen src/f16-f32acc-rdsum/neon.c.in -D CHANNELS_BATCH=64 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-minmax-neonfp16arith-c64.c &

################################## x86 AVX ####################################
tools/xngen src/f16-f32acc-rdsum/avx.c.in -D CHANNELS_BATCH=16 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-f16c-c16.c &
tools/xngen src/f16-f32acc-rdsum/avx.c.in -D CHANNELS_BATCH=32 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-f16c-c32.c &
tools/xngen src/f16-f32acc-rdsum/avx.c.in -D CHANNELS_BATCH=64 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-f16c-c64.c &
tools/xngen src/f16-f32acc-rdsum/avx.c.in -D CHANNELS_BATCH=128 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-f16c-c128.c &

################################## x86 AVX512 #################################
tools/xngen src/f16-f32acc-rdsum/avx512skx.c.in -D CHANNELS_BATCH=16 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c16.c &
tools/xngen src/f16-f32acc-rdsum/avx512skx.c.in -D CHANNELS_BATCH=32 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c32.c &
tools/xngen src/f16-f32acc-rdsum/avx512skx.c.in -D CHANNELS_BATCH=64 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c64.c &
tools/xngen src/f16-f32acc-rdsum/avx512skx.c.in -D CHANNELS_BATCH=128 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c128.c &
tools/xngen src/f16-f32acc-rdsum/avx512skx.c.in -D CHANNELS_BATCH=16 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c16.c &
tools/xngen src/f16-f32acc-rdsum/avx512skx.c.in -D CHANNELS_BATCH=32 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c32.c &
tools/xngen src/f16-f32acc-rdsum/avx512skx.c.in -D CHANNELS_BATCH=64 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c64.c &
tools/xngen src/f16-f32acc-rdsum/avx512skx.c.in -D CHANNELS_BATCH=128 -D ACCUMULATORS=7 -o src/f16-f32acc-rdsum/gen/f16-f32acc-rdsum-7p7x-avx512skx-c128.c &

wait
