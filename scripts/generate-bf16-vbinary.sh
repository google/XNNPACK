#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=ADD     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=DIV     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vdiv-aarch64-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=DIV     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vdiv-aarch64-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MAX     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MIN     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MUL     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=PRELU   -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SQRDIFF -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SUB     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-neon-u16.c &

tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=ADD     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=DIV     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vdivc-aarch64-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=DIV     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vdivc-aarch64-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RDIV    -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrdivc-aarch64-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RDIV    -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrdivc-aarch64-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MAX     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MIN     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MUL     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=PRELU   -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RPRELU  -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RPRELU  -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SQRDIFF -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SUB     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-neon-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RSUB    -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-neon-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=0 -D OP=RSUB    -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-neon-u16.c &

################################# ARM NEON-BF16 ################################
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=ADD     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vadd-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=DIV     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vdiv-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=DIV     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vdiv-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MAX     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmax-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MIN     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmin-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MUL     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vmul-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=PRELU   -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vprelu-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SQRDIFF -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsqrdiff-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SUB     -D BATCH_TILE=8  -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=0 -o src/bf16-vbinary/gen/bf16-vsub-neonbf16-u16.c &

tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=ADD     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=ADD     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vaddc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=DIV     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vdivc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=DIV     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vdivc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=RDIV    -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrdivc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=RDIV    -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrdivc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MAX     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MAX     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmaxc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MIN     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MIN     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vminc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MUL     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=MUL     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vmulc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=PRELU   -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=PRELU   -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vpreluc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=RPRELU  -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=RPRELU  -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrpreluc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SQRDIFF -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SQRDIFF -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsqrdiffc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SUB     -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=SUB     -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vsubc-neonbf16-u16.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=RSUB    -D BATCH_TILE=8  -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-neonbf16-u8.c &
tools/xngen src/bf16-vbinary/neon.c.in -D NEONBF16=1 -D OP=RSUB    -D BATCH_TILE=16 -D BROADCAST=1 -o src/bf16-vbinary/gen/bf16-vrsubc-neonbf16-u16.c &

wait
