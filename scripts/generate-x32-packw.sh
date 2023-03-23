#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x32-packw/scalar.c.in -D NR=2 -D KUNROLL=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x2-scalar-int-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=4 -D KUNROLL=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x4-scalar-int-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=8 -D KUNROLL=4 -D TYPE=uint32_t -o src/x32-packw/gen/x32-packw-x8-scalar-int-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=2 -D KUNROLL=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x2-scalar-float-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=4 -D KUNROLL=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x4-scalar-float-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=8 -D KUNROLL=4 -D TYPE=float    -o src/x32-packw/gen/x32-packw-x8-scalar-float-x4.c &

################################### ARM NEON ##################################
### NR multiple of 4
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=1 -D PREFETCH=0 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8-neon-ld4lane-x4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=1 -D PREFETCH=1 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8-neon-ld4lane-prfm-x4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=12 -D SR=1 -D PREFETCH=0 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x12-neon-ld4lane-x4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=12 -D SR=1 -D PREFETCH=1 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x12-neon-ld4lane-prfm-x4.c &

### SR 4
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=4 -D PREFETCH=0 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8s4-neon-ld4lane-x4.c &
tools/xngen src/x32-packw/neon.c.in -D NR=8  -D SR=4 -D PREFETCH=1 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8s4-neon-ld4lane-prfm-x4.c &

### NR2 micro-kernels
tools/xngen src/x32-packw/NR2-neon.c.in -D NR=2 -D PREFETCH=0 -D KUNROLL=2 -o src/x32-packw/gen/x32-packw-x2-neon-ld2lane-x2.c &
tools/xngen src/x32-packw/NR2-neon.c.in -D NR=2 -D PREFETCH=1 -D KUNROLL=2 -o src/x32-packw/gen/x32-packw-x2-neon-ld2lane-prfm-x2.c &

################################## x86 SSE ##################################
tools/xngen src/x32-packw/sse.c.in -D NR=8  -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x8-sse2-x4.c &
tools/xngen src/x32-packw/sse.c.in -D NR=16 -D KUNROLL=4 -o src/x32-packw/gen/x32-packw-x16-sse2-x4.c &

################################## Unit tests #################################
tools/generate-packw-test.py --spec test/x32-packw.yaml --output test/x32-packw.cc &

wait
