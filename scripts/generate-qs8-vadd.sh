#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### x86 SSE ###################################
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -o src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x8.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -o src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x16.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=2 -o src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x24.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=2 -o src/qs8-vadd/gen/minmax-sse2-mul16-ld64-x32.c

tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -o src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x8.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -o src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x16.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=4 -o src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x24.c
tools/xngen src/qs8-vadd/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=4 -o src/qs8-vadd/gen/minmax-sse41-mul16-ld64-x32.c

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=2 -o src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x8.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=2 -o src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x16.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=2 -o src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x24.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=2 -o src/qs8-vaddc/gen/minmax-sse2-mul16-ld64-x32.c

tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=8  -D SSE=4 -o src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x8.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=16 -D SSE=4 -o src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x16.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=24 -D SSE=4 -o src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x24.c
tools/xngen src/qs8-vaddc/sse-mul16-ld64.c.in -D BATCH_TILE=32 -D SSE=4 -o src/qs8-vaddc/gen/minmax-sse41-mul16-ld64-x32.c

################################## Unit tests #################################
tools/generate-vbinary-test.py --tester VAddMicrokernelTester  --spec test/qs8-vadd-minmax.yaml  --output test/qs8-vadd-minmax.cc
tools/generate-vbinary-test.py --tester VAddCMicrokernelTester --spec test/qs8-vaddc-minmax.yaml --output test/qs8-vaddc-minmax.cc
