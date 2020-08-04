#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### x86 SSE ###################################
tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D SSE=2 -o src/qs8-dwconv/gen/up8x9-minmax-sse2-mul16.c
tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D SSE=2 -o src/qs8-dwconv/gen/up16x9-minmax-sse2-mul16.c
tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9 -D SSE=2 -o src/qs8-dwconv/gen/up24x9-minmax-sse2-mul16.c

tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D SSE=3 -o src/qs8-dwconv/gen/up8x9-minmax-ssse3-mul16.c
tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D SSE=3 -o src/qs8-dwconv/gen/up16x9-minmax-ssse3-mul16.c
tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9 -D SSE=3 -o src/qs8-dwconv/gen/up24x9-minmax-ssse3-mul16.c

tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D SSE=4 -o src/qs8-dwconv/gen/up8x9-minmax-sse41-mul16.c
tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D SSE=4 -o src/qs8-dwconv/gen/up16x9-minmax-sse41-mul16.c
tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9 -D SSE=4 -o src/qs8-dwconv/gen/up24x9-minmax-sse41-mul16.c

tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D SSE=5 -o src/qs8-dwconv/gen/up8x9-minmax-xop-mul16.c
tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D SSE=5 -o src/qs8-dwconv/gen/up16x9-minmax-xop-mul16.c
tools/xngen src/qs8-dwconv/up-sse-mul16.c.in -D CHANNEL_TILE=24 -D KERNEL_TILE=9 -D SSE=5 -o src/qs8-dwconv/gen/up24x9-minmax-xop-mul16.c

################################## Unit tests #################################
tools/generate-dwconv-test.py --spec test/qs8-dwconv-minmax.yaml --output test/qs8-dwconv-minmax.cc
