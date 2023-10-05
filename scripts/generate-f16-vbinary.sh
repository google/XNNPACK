#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=ADD     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=DIV     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-aarch64-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-aarch64-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MUL     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SUB     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vop-neonfp16arith.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-neonfp16arith-u16.c &

tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=ADD      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=ADD      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=DIV      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-aarch64-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=DIV      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-aarch64-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RDIV     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-aarch64-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RDIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-aarch64-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MAX      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MAX      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MIN      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MIN      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MUL      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=MUL      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SQRDIFF  -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SQRDIFF  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SUB      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=SUB      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-neonfp16arith-u16.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RSUB     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-neonfp16arith-u8.c &
tools/xngen src/f16-vbinary/vopc-neonfp16arith.c.in -D OP=RSUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-neonfp16arith-u16.c &

################################### ARM FP16 ##################################
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vop-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-fp16arith-u4.c &

tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=ADD     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=DIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RDIV    -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RDIV    -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RDIV    -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MAX     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MIN     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=MUL     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=1 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=2 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=SUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-fp16arith-u4.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RSUB    -D BATCH_TILE=1 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-fp16arith-u1.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RSUB    -D BATCH_TILE=2 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-fp16arith-u2.c &
tools/xngen src/f16-vbinary/vopc-fp16arith.c.in -D OP=RSUB    -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-fp16arith-u4.c &

################################### x86 F16C ##################################
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=ADD      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=ADD      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vadd-minmax-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=DIV      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=DIV      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdiv-minmax-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MAX      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MAX      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmax-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MIN      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MIN      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmin-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MUL      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=MUL      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmul-minmax-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SQRDIFF  -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SQRDIFF  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiff-f16c-u16.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SUB      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vop-f16c.c.in -D OP=SUB      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsub-minmax-f16c-u16.c &

tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=ADD     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vaddc-minmax-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=DIV     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vdivc-minmax-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RDIV    -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RDIV    -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrdivc-minmax-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vmaxc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vminc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MUL     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vmulc-minmax-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f16-vbinary/gen/f16-vsqrdiffc-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SUB     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vsubc-minmax-f16c-u16.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RSUB    -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-f16c-u8.c &
tools/xngen src/f16-vbinary/vopc-f16c.c.in -D OP=RSUB    -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f16-vbinary/gen/f16-vrsubc-minmax-f16c-u16.c &

wait
