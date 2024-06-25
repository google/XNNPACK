#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-scalar-u8.c &

tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-scalar-u8.c &

tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vadd-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vadd-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vadd-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vadd-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdiv-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdiv-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdiv-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdiv-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmul-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmul-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmul-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmul-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsub-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsub-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsub-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsub-scalar-u8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-scalar-u8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-scalar-u8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vaddc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vaddc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vaddc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vaddc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdivc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdivc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdivc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdivc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmulc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmulc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmulc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmulc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrdivc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrdivc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrdivc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrdivc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrsubc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrsubc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrsubc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrsubc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF  -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF  -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF  -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF  -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsubc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsubc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsubc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsubc-scalar-u8.c &

### WAsm-specific micro-kernels
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-wasm-u8.c &

tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-wasm-u8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-wasm-u8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-wasm-u8.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-x86-u16.c &

tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vadd-relu-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vadd-relu-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vadd-relu-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-relu-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-relu-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-relu-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmul-relu-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmul-relu-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmul-relu-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsub-relu-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsub-relu-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsub-relu-wasmsimd-u16.c &

tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vadd-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vadd-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vadd-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmul-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmul-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmul-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiff-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiff-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiff-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsub-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsub-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsub-wasmsimd-u16.c &

tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-x86-u16.c &

tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-relu-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-relu-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-relu-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-relu-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-relu-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-relu-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-relu-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-relu-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-relu-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-relu-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-relu-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-relu-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-relu-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-relu-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-relu-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-relu-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-relu-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-relu-wasmsimd-u16.c &

tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV    -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV    -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV    -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB    -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB    -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB    -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiffc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiffc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiffc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-wasmsimd-u16.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=ADD     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=ADD     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=DIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-aarch64-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=DIV     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-aarch64-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MAX     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MAX     -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MIN     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MIN     -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SQRDIFF -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-neon-u8.c &

tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=DIV      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-aarch64-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=DIV      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-aarch64-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MAX      -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MAX      -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MIN      -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MIN      -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RDIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-aarch64-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RDIV     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-aarch64-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SQRDIFF  -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SQRDIFF  -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-neon-u8.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=ADD     -D LMUL=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=ADD     -D LMUL=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=DIV     -D LMUL=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=DIV     -D LMUL=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MAX     -D LMUL=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MAX     -D LMUL=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MIN     -D LMUL=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MIN     -D LMUL=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MUL     -D LMUL=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MUL     -D LMUL=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=SQRDIFF -D LMUL=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=SQRDIFF -D LMUL=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=SUB     -D LMUL=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=SUB     -D LMUL=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-rvv-u8v.c &

tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=ADD      -D LMUL=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=ADD      -D LMUL=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=DIV      -D LMUL=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=DIV      -D LMUL=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MAX      -D LMUL=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MAX      -D LMUL=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MIN      -D LMUL=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MIN      -D LMUL=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MUL      -D LMUL=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MUL      -D LMUL=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MUL      -D LMUL=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmulc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MUL      -D LMUL=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmulc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=RDIV     -D LMUL=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=RDIV     -D LMUL=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=RSUB     -D LMUL=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=RSUB     -D LMUL=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=SQRDIFF  -D LMUL=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=SQRDIFF  -D LMUL=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=SUB      -D LMUL=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=SUB      -D LMUL=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-rvv-u8v.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=DIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=DIV     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MAX     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MAX     -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MIN     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MIN     -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SQRDIFF -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-sse-u8.c &

tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=DIV      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=DIV      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MAX      -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MAX      -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MIN      -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MIN      -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RDIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RDIV     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SQRDIFF  -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SQRDIFF  -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-sse-u8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=ADD     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=DIV     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MUL     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SUB     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-avx-u8.c &

tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=ADD      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=ADD      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=DIV      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=DIV      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MAX      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MAX      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MIN      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MIN      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MUL      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MUL      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RDIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RDIV     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RSUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RSUB     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SQRDIFF  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SQRDIFF  -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SUB      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SUB      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-avx-u8.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=ADD     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=DIV     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MAX     -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MIN     -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MUL     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SQRDIFF -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SUB     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-avx512f-u32.c &

tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=ADD      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=ADD      -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=DIV      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=DIV      -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MAX      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MAX      -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MIN      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MIN      -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MUL      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MUL      -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RDIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RDIV     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RSUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RSUB     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SQRDIFF  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SQRDIFF  -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SUB      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SUB      -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-avx512f-u32.c &

################################### HEXAGON HVX ##################################
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=ADD     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=ADD     -D BATCH_TILE=64 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-hvx-u64.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=ADD     -D BATCH_TILE=128 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-hvx-u128.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MAX     -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MAX     -D BATCH_TILE=64 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-hvx-u64.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MAX     -D BATCH_TILE=128 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-hvx-u128.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MIN     -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MIN     -D BATCH_TILE=64 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-hvx-u64.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MIN     -D BATCH_TILE=128 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-hvx-u128.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MUL     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MUL     -D BATCH_TILE=64 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-hvx-u64.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MUL     -D BATCH_TILE=128 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-hvx-u128.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SUB     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SUB     -D BATCH_TILE=64 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-hvx-u64.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SUB     -D BATCH_TILE=128 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-hvx-u128.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SQRDIFF     -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SQRDIFF     -D BATCH_TILE=64 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-hvx-u64.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SQRDIFF     -D BATCH_TILE=128 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-hvx-u128.c &

tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=ADD     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=ADD     -D BATCH_TILE=64 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=ADD     -D BATCH_TILE=128 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MAX     -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MAX     -D BATCH_TILE=64 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MAX     -D BATCH_TILE=128 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MIN     -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MIN     -D BATCH_TILE=64 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MIN     -D BATCH_TILE=128 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MUL     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MUL     -D BATCH_TILE=64 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MUL     -D BATCH_TILE=128 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SUB     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SUB     -D BATCH_TILE=64 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SUB     -D BATCH_TILE=128 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=RSUB     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=RSUB     -D BATCH_TILE=64 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=RSUB     -D BATCH_TILE=128 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SQRDIFF     -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SQRDIFF     -D BATCH_TILE=64 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SQRDIFF     -D BATCH_TILE=128 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-hvx-u128.c &

wait
