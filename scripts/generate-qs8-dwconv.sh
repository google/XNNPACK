#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ##################################
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up8x9-minmax-neon-mul16.c
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up16x9-minmax-neon-mul16.c
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up24x9-minmax-neon-mul16.c
tools/xngen src/qs8-dwconv/unipass-neon-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up32x9-minmax-neon-mul16.c

################################## WAsm SIMD ##################################
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up8x9-minmax-wasmsimd-mul16.c
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up16x9-minmax-wasmsimd-mul16.c
tools/xngen src/qs8-dwconv/unipass-wasmsimd-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up24x9-minmax-wasmsimd-mul16.c

################################### x86 SSE ###################################
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D SSE=2 -o src/qs8-dwconv/gen/up8x9-minmax-sse2-mul16.c
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D SSE=2 -o src/qs8-dwconv/gen/up16x9-minmax-sse2-mul16.c
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9 -D SSE=2 -o src/qs8-dwconv/gen/up24x9-minmax-sse2-mul16.c

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D SSE=3 -o src/qs8-dwconv/gen/up8x9-minmax-ssse3-mul16.c
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D SSE=3 -o src/qs8-dwconv/gen/up16x9-minmax-ssse3-mul16.c
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9 -D SSE=3 -o src/qs8-dwconv/gen/up24x9-minmax-ssse3-mul16.c

tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D SSE=4 -o src/qs8-dwconv/gen/up8x9-minmax-sse41-mul16.c
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D SSE=4 -o src/qs8-dwconv/gen/up16x9-minmax-sse41-mul16.c
tools/xngen src/qs8-dwconv/unipass-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9 -D SSE=4 -o src/qs8-dwconv/gen/up24x9-minmax-sse41-mul16.c

################################### x86 AVX2 ##################################
tools/xngen src/qs8-dwconv/unipass-avx2-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up16x9-minmax-avx2-mul16.c
tools/xngen src/qs8-dwconv/unipass-avx2-mul16.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up32x9-minmax-avx2-mul16.c

tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up8x9-minmax-avx2-mul32.c
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up16x9-minmax-avx2-mul32.c
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up24x9-minmax-avx2-mul32.c
tools/xngen src/qs8-dwconv/unipass-avx2-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up32x9-minmax-avx2-mul32.c

################################## x86 AVX512 #################################
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up16x9-minmax-avx512skx-mul32.c
tools/xngen src/qs8-dwconv/unipass-avx512skx-mul32.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -o src/qs8-dwconv/gen/up32x9-minmax-avx512skx-mul32.c

################################## Unit tests #################################
tools/generate-dwconv-test.py --spec test/qs8-dwconv-minmax.yaml --output test/qs8-dwconv-minmax.cc
