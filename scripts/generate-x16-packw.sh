#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/x32-packw/scalar.c.in -D NR=8  -D KUNROLL=4 -D TYPE=uint16_t -o src/x16-packw/gen/x16-packw-x8-scalar-int-x4.c &
tools/xngen src/x32-packw/scalar.c.in -D NR=16 -D KUNROLL=4 -D TYPE=uint16_t -o src/x16-packw/gen/x16-packw-x16-scalar-int-x4.c &

################################### ARM NEON ##################################
### NR multiple of 4
tools/xngen src/x16-packw/neon.c.in -D NR=8  -D PREFETCH=0 -D KUNROLL=4 -o src/x16-packw/gen/x16-packw-x8-neon-ld4lane-x4.c &
tools/xngen src/x16-packw/neon.c.in -D NR=8  -D PREFETCH=1 -D KUNROLL=4 -o src/x16-packw/gen/x16-packw-x8-neon-ld4lane-prfm-x4.c &
tools/xngen src/x16-packw/neon.c.in -D NR=16 -D PREFETCH=0 -D KUNROLL=4 -o src/x16-packw/gen/x16-packw-x16-neon-ld4lane-x4.c &
tools/xngen src/x16-packw/neon.c.in -D NR=16 -D PREFETCH=1 -D KUNROLL=4 -o src/x16-packw/gen/x16-packw-x16-neon-ld4lane-prfm-x4.c &
tools/xngen src/x16-packw/neon.c.in -D NR=8  -D PREFETCH=0 -D KUNROLL=8 -o src/x16-packw/gen/x16-packw-x8-neon-ld4lane-x8.c &
tools/xngen src/x16-packw/neon.c.in -D NR=8  -D PREFETCH=1 -D KUNROLL=8 -o src/x16-packw/gen/x16-packw-x8-neon-ld4lane-prfm-x8.c &
tools/xngen src/x16-packw/neon.c.in -D NR=16 -D PREFETCH=0 -D KUNROLL=8 -o src/x16-packw/gen/x16-packw-x16-neon-ld4lane-x8.c &
tools/xngen src/x16-packw/neon.c.in -D NR=16 -D PREFETCH=1 -D KUNROLL=8 -o src/x16-packw/gen/x16-packw-x16-neon-ld4lane-prfm-x8.c &
tools/xngen src/x16-packw/neon.c.in -D NR=8  -D PREFETCH=0 -D KUNROLL=12 -o src/x16-packw/gen/x16-packw-x8-neon-ld4lane-x12.c &
tools/xngen src/x16-packw/neon.c.in -D NR=8  -D PREFETCH=1 -D KUNROLL=12 -o src/x16-packw/gen/x16-packw-x8-neon-ld4lane-prfm-x12.c &
tools/xngen src/x16-packw/neon.c.in -D NR=16 -D PREFETCH=0 -D KUNROLL=12 -o src/x16-packw/gen/x16-packw-x16-neon-ld4lane-x12.c &
tools/xngen src/x16-packw/neon.c.in -D NR=16 -D PREFETCH=1 -D KUNROLL=12 -o src/x16-packw/gen/x16-packw-x16-neon-ld4lane-prfm-x12.c &
tools/xngen src/x16-packw/neon.c.in -D NR=8  -D PREFETCH=0 -D KUNROLL=16 -o src/x16-packw/gen/x16-packw-x8-neon-ld4lane-x16.c &
tools/xngen src/x16-packw/neon.c.in -D NR=8  -D PREFETCH=1 -D KUNROLL=16 -o src/x16-packw/gen/x16-packw-x8-neon-ld4lane-prfm-x16.c &
tools/xngen src/x16-packw/neon.c.in -D NR=16 -D PREFETCH=0 -D KUNROLL=16 -o src/x16-packw/gen/x16-packw-x16-neon-ld4lane-x16.c &
tools/xngen src/x16-packw/neon.c.in -D NR=16 -D PREFETCH=1 -D KUNROLL=16 -o src/x16-packw/gen/x16-packw-x16-neon-ld4lane-prfm-x16.c &

################################### X86 AVX2 ##################################
tools/xngen src/x16-packw/avx.c.in -D NR=8 -D PREFETCH=0 -D KUNROLL=16 -o src/x16-packw/gen/x16-packw-x8-avx2-x16.c &
tools/xngen src/x16-packw/avx.c.in -D NR=8 -D PREFETCH=1 -D KUNROLL=16 -o src/x16-packw/gen/x16-packw-x8-avx2-prfm-x16.c &
tools/xngen src/x16-packw/avx.c.in -D NR=16 -D PREFETCH=0 -D KUNROLL=16 -o src/x16-packw/gen/x16-packw-x16-avx2-x16.c &
tools/xngen src/x16-packw/avx.c.in -D NR=16 -D PREFETCH=1 -D KUNROLL=16 -o src/x16-packw/gen/x16-packw-x16-avx2-prfm-x16.c &

################################## Unit tests #################################
tools/generate-packw-test.py --spec test/x16-packw.yaml --output test/x16-packw.cc &

wait
