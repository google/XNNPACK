#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-prelu/scalar.c.in -D CHANNEL_TILE=1 -D ROW_TILE=2 -D WASM=0 -o src/f32-prelu/gen/scalar-2x1.c
tools/xngen src/f32-prelu/scalar.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D WASM=0 -o src/f32-prelu/gen/scalar-2x4.c

################################### ARM NEON ##################################
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -o src/f32-prelu/gen/neon-2x4.c
tools/xngen src/f32-prelu/neon.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -o src/f32-prelu/gen/neon-2x8.c

#################################### PSIMD ####################################
tools/xngen src/f32-prelu/psimd.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -o src/f32-prelu/gen/psimd-2x4.c
tools/xngen src/f32-prelu/psimd.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -o src/f32-prelu/gen/psimd-2x8.c

################################### x86 SSE2 ##################################
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D BLEND=0 -o src/f32-prelu/gen/sse2-2x4.c
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -D BLEND=0 -o src/f32-prelu/gen/sse2-2x8.c
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D BLEND=1 -o src/f32-prelu/gen/sse41-2x4.c
tools/xngen src/f32-prelu/sse.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -D BLEND=1 -o src/f32-prelu/gen/sse41-2x8.c

################################### x86 AVX ###################################
tools/xngen src/f32-prelu/avx.c.in -D CHANNEL_TILE=8  -D ROW_TILE=2 -o src/f32-prelu/gen/avx-2x8.c
tools/xngen src/f32-prelu/avx.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f32-prelu/gen/avx-2x16.c

################################## x86 AVX512 #################################
tools/xngen src/f32-prelu/avx512f.c.in -D CHANNEL_TILE=16 -D ROW_TILE=2 -o src/f32-prelu/gen/avx512f-2x16.c
tools/xngen src/f32-prelu/avx512f.c.in -D CHANNEL_TILE=32 -D ROW_TILE=2 -o src/f32-prelu/gen/avx512f-2x32.c

################################## Unit tests #################################
tools/generate-prelu-test.py --spec test/f32-prelu.yaml --output test/f32-prelu.cc
