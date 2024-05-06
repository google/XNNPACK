#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-rdsum/scalar.c.in -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-scalar.c &

#################################### NEON ###################################
tools/xngen src/f32-rdsum/neon.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-c16.c &
tools/xngen src/f32-rdsum/neon.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-c32.c &
tools/xngen src/f32-rdsum/neon.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-c64.c &

#################################### SSE ####################################
tools/xngen src/f32-rdsum/sse.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse-c16.c &
tools/xngen src/f32-rdsum/sse.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse-c32.c &
tools/xngen src/f32-rdsum/sse.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse-c64.c &

#################################### AVX ####################################
tools/xngen src/f32-rdsum/avx.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c16.c &
tools/xngen src/f32-rdsum/avx.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c32.c &
tools/xngen src/f32-rdsum/avx.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c64.c &

wait
