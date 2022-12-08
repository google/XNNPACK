#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=ADD     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=DIV     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-aarch64-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-aarch64-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MUL     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SUB     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-neonfp16arith-x16.c &

tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=ADD      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=ADD      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=DIV      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-aarch64-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=DIV      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-aarch64-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RDIV     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-aarch64-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RDIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-aarch64-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MAX      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MAX      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MIN      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MIN      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MUL      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MUL      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SQRDIFF  -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SQRDIFF  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SUB      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SUB      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-neonfp16arith-x16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RSUB     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-neonfp16arith-x8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RSUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-neonfp16arith-x16.c &

################################### ARM FP16 ##################################
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-fp16arith-x4.c &

tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RDIV    -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RDIV    -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RDIV    -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-fp16arith-x4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RSUB    -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-fp16arith-x1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RSUB    -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-fp16arith-x2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RSUB    -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-fp16arith-x4.c &

################################### x86 F16C ##################################
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=ADD      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-f16c-x8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=ADD      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-f16c-x16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=DIV      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-f16c-x8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=DIV      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-f16c-x16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MAX      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-f16c-x8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MAX      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-f16c-x16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MIN      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-f16c-x8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MIN      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-f16c-x16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MUL      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-f16c-x8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MUL      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-f16c-x16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SQRDIFF  -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-f16c-x8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SQRDIFF  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-f16c-x16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SUB      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-f16c-x8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SUB      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-f16c-x16.c &

tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=ADD     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-f16c-x8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-f16c-x16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=DIV     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-f16c-x8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-f16c-x16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RDIV    -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-f16c-x8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RDIV    -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-f16c-x16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-f16c-x8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-f16c-x16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-f16c-x8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-f16c-x16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MUL     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-f16c-x8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-f16c-x16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-f16c-x8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-f16c-x16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SUB     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-f16c-x8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-f16c-x16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RSUB    -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-f16c-x8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RSUB    -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-f16c-x16.c &

################################## Unit tests #################################
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vadd-minmax.yaml --output test/f16-vadd-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vdiv-minmax.yaml --output test/f16-vdiv-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vmax.yaml --output test/f16-vmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vmin.yaml --output test/f16-vmin.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vmul-minmax.yaml --output test/f16-vmul-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vsqrdiff.yaml --output test/f16-vsqrdiff.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f16-vsub-minmax.yaml --output test/f16-vsub-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vaddc-minmax.yaml --output test/f16-vaddc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vdivc-minmax.yaml --output test/f16-vdivc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vrdivc-minmax.yaml --output test/f16-vrdivc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vmaxc.yaml --output test/f16-vmaxc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vminc.yaml --output test/f16-vminc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vmulc-minmax.yaml --output test/f16-vmulc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vsqrdiffc.yaml --output test/f16-vsqrdiffc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vsubc-minmax.yaml --output test/f16-vsubc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f16-vrsubc-minmax.yaml --output test/f16-vrsubc-minmax.cc &

wait
