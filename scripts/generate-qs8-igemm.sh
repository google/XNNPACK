#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### x86 SSE ###################################
### C2 micro-kernels
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=1 -D SSE=2 -D LD128=0 -o src/qs8-igemm/gen/1x4c2-minmax-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=4 -D SSE=2 -D LD128=0 -o src/qs8-igemm/gen/4x4c2-minmax-sse2-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=1 -D SSE=3 -D LD128=0 -o src/qs8-igemm/gen/1x4c2-minmax-ssse3-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=4 -D SSE=3 -D LD128=0 -o src/qs8-igemm/gen/4x4c2-minmax-ssse3-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=1 -D SSE=4 -D LD128=0 -o src/qs8-igemm/gen/1x4c2-minmax-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=4 -D SSE=4 -D LD128=0 -o src/qs8-igemm/gen/4x4c2-minmax-sse41-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=1 -D SSE=5 -D LD128=0 -o src/qs8-igemm/gen/1x4c2-minmax-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=4 -D SSE=5 -D LD128=0 -o src/qs8-igemm/gen/4x4c2-minmax-xop-ld64.c

tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=1 -D SSE=2 -D LD128=1 -o src/qs8-igemm/gen/1x4c2-minmax-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=4 -D SSE=2 -D LD128=1 -o src/qs8-igemm/gen/4x4c2-minmax-sse2-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=1 -D SSE=3 -D LD128=1 -o src/qs8-igemm/gen/1x4c2-minmax-ssse3-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=4 -D SSE=3 -D LD128=1 -o src/qs8-igemm/gen/4x4c2-minmax-ssse3-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=1 -D SSE=4 -D LD128=1 -o src/qs8-igemm/gen/1x4c2-minmax-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=4 -D SSE=4 -D LD128=1 -o src/qs8-igemm/gen/4x4c2-minmax-sse41-ld128.c

tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=1 -D SSE=5 -D LD128=1 -o src/qs8-igemm/gen/1x4c2-minmax-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=4 -D SSE=5 -D LD128=1 -o src/qs8-igemm/gen/4x4c2-minmax-xop-ld128.c

### C8 micro-kernels
tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=1 -D SSE=2 -D LD128=0 -o src/qs8-igemm/gen/1x4c8-minmax-sse2-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=2 -D SSE=2 -D LD128=0 -o src/qs8-igemm/gen/2x4c8-minmax-sse2-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=1 -D SSE=3 -D LD128=0 -o src/qs8-igemm/gen/1x4c8-minmax-ssse3-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=2 -D SSE=3 -D LD128=0 -o src/qs8-igemm/gen/2x4c8-minmax-ssse3-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=1 -D SSE=4 -D LD128=0 -o src/qs8-igemm/gen/1x4c8-minmax-sse41-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=2 -D SSE=4 -D LD128=0 -o src/qs8-igemm/gen/2x4c8-minmax-sse41-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=1 -D SSE=5 -D LD128=0 -o src/qs8-igemm/gen/1x4c8-minmax-xop-ld64.c
tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=2 -D SSE=5 -D LD128=0 -o src/qs8-igemm/gen/2x4c8-minmax-xop-ld64.c

tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=1 -D SSE=2 -D LD128=1 -o src/qs8-igemm/gen/1x4c8-minmax-sse2-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=2 -D SSE=2 -D LD128=1 -o src/qs8-igemm/gen/2x4c8-minmax-sse2-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=1 -D SSE=3 -D LD128=1 -o src/qs8-igemm/gen/1x4c8-minmax-ssse3-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=2 -D SSE=3 -D LD128=1 -o src/qs8-igemm/gen/2x4c8-minmax-ssse3-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=1 -D SSE=4 -D LD128=1 -o src/qs8-igemm/gen/1x4c8-minmax-sse41-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=2 -D SSE=4 -D LD128=1 -o src/qs8-igemm/gen/2x4c8-minmax-sse41-ld128.c

tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=1 -D SSE=5 -D LD128=1 -o src/qs8-igemm/gen/1x4c8-minmax-xop-ld128.c
tools/xngen src/qs8-igemm/MRx4c8-minmax-sse.c.in -D MR=2 -D SSE=5 -D LD128=1 -o src/qs8-igemm/gen/2x4c8-minmax-xop-ld128.c

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/qs8-igemm-minmax.yaml --output test/qs8-igemm-minmax.cc
