#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### x86 SSE ###################################
### C2 micro-kernels
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=1 -D SSE=2 -o src/qs8-igemm/gen/1x4c2-minmax-sse2.c
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=4 -D SSE=2 -o src/qs8-igemm/gen/4x4c2-minmax-sse2.c

tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=1 -D SSE=3 -o src/qs8-igemm/gen/1x4c2-minmax-ssse3.c
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=4 -D SSE=3 -o src/qs8-igemm/gen/4x4c2-minmax-ssse3.c

tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=1 -D SSE=4 -o src/qs8-igemm/gen/1x4c2-minmax-sse41.c
tools/xngen src/qs8-igemm/MRx4c2-minmax-sse.c.in -D MR=4 -D SSE=4 -o src/qs8-igemm/gen/4x4c2-minmax-sse41.c

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/qs8-igemm-minmax.yaml --output test/qs8-igemm-minmax.cc
