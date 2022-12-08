#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-scalar-x8.c &

tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-scalar-x8.c &

tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vadd-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vadd-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vadd-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vadd-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdiv-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdiv-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdiv-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdiv-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmul-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmul-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmul-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmul-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsub-scalar-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsub-scalar-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsub-scalar-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsub-scalar-x8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-scalar-x8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-scalar-x8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vaddc-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vaddc-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vaddc-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vaddc-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdivc-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdivc-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdivc-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vdivc-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmulc-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmulc-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmulc-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmulc-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrdivc-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrdivc-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrdivc-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrdivc-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrsubc-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrsubc-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrsubc-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB     -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vrsubc-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF  -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF  -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF  -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF  -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsubc-scalar-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsubc-scalar-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsubc-scalar-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB      -D BATCH_TILE=8 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsubc-scalar-x8.c &

### WAsm-specific micro-kernels
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-wasm-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-wasm-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-wasm-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-wasm-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-wasm-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-wasm-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-wasm-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-wasm-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-wasm-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-wasm-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-wasm-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-wasm-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-wasm-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-wasm-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-wasm-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-wasm-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-wasm-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-wasm-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-wasm-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-wasm-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-wasm-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-wasm-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-wasm-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-wasm-x8.c &

tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-wasm-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-wasm-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-wasm-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vadd-relu-wasm-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-wasm-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-wasm-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-wasm-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdiv-relu-wasm-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-wasm-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-wasm-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-wasm-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmul-relu-wasm-x8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-wasm-x1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-wasm-x2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-wasm-x4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsub-relu-wasm-x8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-wasm-x8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vaddc-relu-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vdivc-relu-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vmulc-relu-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrdivc-relu-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vrsubc-relu-wasm-x8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-wasm-x1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-wasm-x2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-wasm-x4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=8 -D WASM=1 -D ACTIVATION=RELU -o src/f32-vbinary/gen/f32-vsubc-relu-wasm-x8.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vadd-minmax-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdiv-minmax-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmul-minmax-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsub-minmax-wasmsimd-x86-x16.c &

tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vadd-relu-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vadd-relu-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vadd-relu-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-relu-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-relu-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-relu-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmul-relu-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmul-relu-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmul-relu-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsub-relu-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsub-relu-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsub-relu-wasmsimd-x16.c &

tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vadd-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vadd-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vadd-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmul-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmul-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmul-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiff-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiff-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiff-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsub-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsub-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsub-wasmsimd-x16.c &

tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vaddc-minmax-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vdivc-minmax-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vmulc-minmax-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrdivc-minmax-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vrsubc-minmax-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=4  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -D X86=1 -o src/f32-vbinary/gen/f32-vsubc-minmax-wasmsimd-x86-x16.c &

tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-relu-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-relu-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD  -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-relu-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-relu-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-relu-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV  -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-relu-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-relu-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-relu-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL  -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-relu-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-relu-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-relu-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-relu-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-relu-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-relu-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-relu-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=4  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-relu-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=8  -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-relu-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB  -D BATCH_TILE=16 -D ACTIVATION=RELU -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-relu-wasmsimd-x16.c &

tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-arm-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-x86-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-arm-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-x86-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-arm-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=1 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-x86-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV    -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV    -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV    -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB    -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB    -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB    -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiffc-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiffc-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiffc-wasmsimd-x16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=4  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-wasmsimd-x4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-wasmsimd-x8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-wasmsimd-x16.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=ADD     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-neon-x4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=ADD     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-neon-x8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=DIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-aarch64-neon-x4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=DIV     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-aarch64-neon-x8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MAX     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-neon-x4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MAX     -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-neon-x8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MIN     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-neon-x4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MIN     -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-neon-x8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-neon-x4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-neon-x8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-neon-x4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SQRDIFF -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-neon-x8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-neon-x4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-neon-x8.c &

tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-neon-x4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-neon-x8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=DIV      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-aarch64-neon-x4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=DIV      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-aarch64-neon-x8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MAX      -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-neon-x4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MAX      -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-neon-x8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MIN      -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-neon-x4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MIN      -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-neon-x8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-neon-x4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-neon-x8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RDIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-aarch64-neon-x4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RDIV     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-aarch64-neon-x8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-neon-x4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-neon-x8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SQRDIFF  -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-neon-x4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SQRDIFF  -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-neon-x8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-neon-x4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-neon-x8.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-sse-x4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-sse-x8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=DIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-sse-x4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=DIV     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-sse-x8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MAX     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-sse-x4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MAX     -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-sse-x8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MIN     -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-sse-x4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MIN     -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-sse-x8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-sse-x4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-sse-x8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SQRDIFF -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-sse-x4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SQRDIFF -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-sse-x8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-sse-x4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-sse-x8.c &

tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-sse-x4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-sse-x8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=DIV      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-sse-x4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=DIV      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-sse-x8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MAX      -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-sse-x4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MAX      -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-sse-x8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MIN      -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-sse-x4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MIN      -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-sse-x8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-sse-x4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-sse-x8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RDIV     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-sse-x4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RDIV     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-sse-x8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB     -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-sse-x4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB     -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-sse-x8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SQRDIFF  -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-sse-x4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SQRDIFF  -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-sse-x8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB      -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-sse-x4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB      -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-sse-x8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-avx-x16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=ADD     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-avx-x8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-avx-x16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=DIV     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-avx-x8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-avx-x16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MAX     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-avx-x8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-avx-x16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MIN     -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-avx-x8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-avx-x16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MUL     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-avx-x8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-avx-x16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SQRDIFF -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-avx-x8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-avx-x16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SUB     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-avx-x8.c &

tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=ADD      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-avx-x16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=ADD      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-avx-x8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=DIV      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-avx-x16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=DIV      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-avx-x8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MAX      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-avx-x16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MAX      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-avx-x8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MIN      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-avx-x16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MIN      -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-avx-x8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MUL      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-avx-x16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MUL      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-avx-x8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RDIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-avx-x16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RDIV     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-avx-x8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RSUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-avx-x16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RSUB     -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-avx-x8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SQRDIFF  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-avx-x16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SQRDIFF  -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-avx-x8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SUB      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-avx-x16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SUB      -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-avx-x8.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=ADD     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-avx512f-x16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=ADD     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vadd-minmax-avx512f-x32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=DIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-avx512f-x16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=DIV     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdiv-minmax-avx512f-x32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MAX     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-avx512f-x16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MAX     -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmax-avx512f-x32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MIN     -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-avx512f-x16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MIN     -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmin-avx512f-x32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MUL     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-avx512f-x16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MUL     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmul-minmax-avx512f-x32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SQRDIFF -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-avx512f-x16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SQRDIFF -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiff-avx512f-x32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-avx512f-x16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SUB     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsub-minmax-avx512f-x32.c &

tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=ADD      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-avx512f-x16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=ADD      -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vaddc-minmax-avx512f-x32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=DIV      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-avx512f-x16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=DIV      -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vdivc-minmax-avx512f-x32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MAX      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-avx512f-x16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MAX      -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vmaxc-avx512f-x32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MIN      -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-avx512f-x16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MIN      -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vminc-avx512f-x32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MUL      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-avx512f-x16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MUL      -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vmulc-minmax-avx512f-x32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RDIV     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-avx512f-x16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RDIV     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrdivc-minmax-avx512f-x32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RSUB     -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-avx512f-x16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RSUB     -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vrsubc-minmax-avx512f-x32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SQRDIFF  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-avx512f-x16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SQRDIFF  -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/f32-vsqrdiffc-avx512f-x32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SUB      -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-avx512f-x16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SUB      -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/f32-vsubc-minmax-avx512f-x32.c &

################################## Unit tests #################################
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vadd-minmax.yaml --output test/f32-vadd-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vadd-relu.yaml   --output test/f32-vadd-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vadd.yaml        --output test/f32-vadd.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vdiv-minmax.yaml --output test/f32-vdiv-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vdiv-relu.yaml   --output test/f32-vdiv-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vdiv.yaml        --output test/f32-vdiv.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vmax.yaml        --output test/f32-vmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vmin.yaml        --output test/f32-vmin.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vmul-minmax.yaml --output test/f32-vmul-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vmul-relu.yaml   --output test/f32-vmul-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vmul.yaml        --output test/f32-vmul.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vsqrdiff.yaml    --output test/f32-vsqrdiff.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vsub-minmax.yaml --output test/f32-vsub-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vsub-relu.yaml   --output test/f32-vsub-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryMicrokernelTester  --spec test/f32-vsub.yaml        --output test/f32-vsub.cc &

tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vaddc-minmax.yaml  --output test/f32-vaddc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vaddc-relu.yaml    --output test/f32-vaddc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vaddc.yaml         --output test/f32-vaddc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vdivc-minmax.yaml  --output test/f32-vdivc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vdivc-relu.yaml    --output test/f32-vdivc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vdivc.yaml         --output test/f32-vdivc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vmaxc.yaml         --output test/f32-vmaxc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vminc.yaml         --output test/f32-vminc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vmulc-minmax.yaml  --output test/f32-vmulc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vmulc-relu.yaml    --output test/f32-vmulc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vmulc.yaml         --output test/f32-vmulc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrdivc-minmax.yaml --output test/f32-vrdivc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrdivc-relu.yaml   --output test/f32-vrdivc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrdivc.yaml        --output test/f32-vrdivc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrsubc-minmax.yaml --output test/f32-vrsubc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrsubc-relu.yaml   --output test/f32-vrsubc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vrsubc.yaml        --output test/f32-vrsubc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vsqrdiffc.yaml     --output test/f32-vsqrdiffc.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vsubc-minmax.yaml  --output test/f32-vsubc-minmax.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vsubc-relu.yaml    --output test/f32-vsubc-relu.cc &
tools/generate-vbinary-test.py --tester VBinaryCMicrokernelTester --spec test/f32-vsubc.yaml         --output test/f32-vsubc.cc &

wait
