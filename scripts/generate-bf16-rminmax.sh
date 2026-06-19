#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D OP=MINMAX -o src/bf16-rminmax/gen/bf16-rminmax-neon-u8.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MINMAX -o src/bf16-rminmax/gen/bf16-rminmax-neon-u16-acc2.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -D OP=MINMAX -o src/bf16-rminmax/gen/bf16-rminmax-neon-u24-acc3.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MINMAX -o src/bf16-rminmax/gen/bf16-rminmax-neon-u32-acc2.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -D OP=MINMAX -o src/bf16-rminmax/gen/bf16-rminmax-neon-u32-acc4.c &

tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D OP=MAX -o src/bf16-rminmax/gen/bf16-rmax-neon-u8.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MAX -o src/bf16-rminmax/gen/bf16-rmax-neon-u16-acc2.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -D OP=MAX -o src/bf16-rminmax/gen/bf16-rmax-neon-u24-acc3.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MAX -o src/bf16-rminmax/gen/bf16-rmax-neon-u32-acc2.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -D OP=MAX -o src/bf16-rminmax/gen/bf16-rmax-neon-u32-acc4.c &

tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D OP=MIN -o src/bf16-rminmax/gen/bf16-rmin-neon-u8.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MIN -o src/bf16-rminmax/gen/bf16-rmin-neon-u16-acc2.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -D OP=MIN -o src/bf16-rminmax/gen/bf16-rmin-neon-u24-acc3.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MIN -o src/bf16-rminmax/gen/bf16-rmin-neon-u32-acc2.c &
tools/xngen src/bf16-rminmax/neon.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -D OP=MIN -o src/bf16-rminmax/gen/bf16-rmin-neon-u32-acc4.c &

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MINMAX -o src/bf16-rminmax/gen/bf16-rminmax-scalar-u1.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MINMAX -o src/bf16-rminmax/gen/bf16-rminmax-scalar-u2-acc2.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MINMAX -o src/bf16-rminmax/gen/bf16-rminmax-scalar-u3-acc3.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MINMAX -o src/bf16-rminmax/gen/bf16-rminmax-scalar-u4-acc2.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MINMAX -o src/bf16-rminmax/gen/bf16-rminmax-scalar-u4-acc4.c &

tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MAX -o src/bf16-rminmax/gen/bf16-rmax-scalar-u1.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MAX -o src/bf16-rminmax/gen/bf16-rmax-scalar-u2-acc2.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MAX -o src/bf16-rminmax/gen/bf16-rmax-scalar-u3-acc3.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MAX -o src/bf16-rminmax/gen/bf16-rmax-scalar-u4-acc2.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MAX -o src/bf16-rminmax/gen/bf16-rmax-scalar-u4-acc4.c &

tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MIN -o src/bf16-rminmax/gen/bf16-rmin-scalar-u1.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MIN -o src/bf16-rminmax/gen/bf16-rmin-scalar-u2-acc2.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MIN -o src/bf16-rminmax/gen/bf16-rmin-scalar-u3-acc3.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MIN -o src/bf16-rminmax/gen/bf16-rmin-scalar-u4-acc2.c &
tools/xngen src/bf16-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MIN -o src/bf16-rminmax/gen/bf16-rmin-scalar-u4-acc4.c &

wait
