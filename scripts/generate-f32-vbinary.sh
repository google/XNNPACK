#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD          -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vadd-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD          -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vadd-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD          -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vadd-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD          -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vadd-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV          -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vdiv-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV          -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vdiv-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV          -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vdiv-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV          -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vdiv-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX          -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmax-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX          -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmax-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX          -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmax-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX          -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmax-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN          -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmin-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN          -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmin-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN          -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmin-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN          -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmin-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL          -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmul-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL          -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmul-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL          -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmul-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL          -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmul-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=PRELU        -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vprelu-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=PRELU        -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vprelu-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=PRELU        -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vprelu-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=PRELU        -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vprelu-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF      -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF      -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF      -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SQRDIFF      -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsqrdiff-scalar-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB          -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsub-scalar-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB          -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsub-scalar-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB          -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsub-scalar-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB          -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsub-scalar-u8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD         -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vaddc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD         -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vaddc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD         -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vaddc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD         -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vaddc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV         -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vdivc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV         -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vdivc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV         -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vdivc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV         -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vdivc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX         -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmaxc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX         -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmaxc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX         -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmaxc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX         -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmaxc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN         -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vminc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN         -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vminc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN         -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vminc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN         -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vminc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL         -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmulc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL         -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmulc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL         -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmulc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL         -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vmulc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=PRELU       -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vpreluc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=PRELU       -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vpreluc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=PRELU       -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vpreluc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=PRELU       -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vpreluc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV        -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrdivc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV        -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrdivc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV        -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrdivc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV        -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrdivc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RPRELU      -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrpreluc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RPRELU      -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrpreluc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RPRELU      -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrpreluc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RPRELU      -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrpreluc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB        -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrsubc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB        -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrsubc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB        -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrsubc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB        -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vrsubc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF     -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF     -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF     -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SQRDIFF     -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-scalar-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB         -D BATCH_TILE=1 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsubc-scalar-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB         -D BATCH_TILE=2 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsubc-scalar-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB         -D BATCH_TILE=4 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsubc-scalar-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB         -D BATCH_TILE=8 -D WASM=0 -D -o src/f32-vbinary/gen/f32-vsubc-scalar-u8.c &

### WAsm-specific micro-kernels
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD          -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vadd-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD          -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vadd-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD          -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vadd-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD          -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vadd-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV          -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vdiv-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV          -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vdiv-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV          -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vdiv-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV          -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vdiv-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX          -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmax-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX          -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmax-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX          -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmax-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX          -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmax-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN          -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmin-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN          -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmin-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN          -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmin-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN          -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmin-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL          -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmul-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL          -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmul-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL          -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmul-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL          -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmul-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=PRELU        -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vprelu-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=PRELU        -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vprelu-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=PRELU        -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vprelu-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=PRELU        -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vprelu-wasm-u8.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB          -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vsub-wasm-u1.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB          -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vsub-wasm-u2.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB          -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vsub-wasm-u4.c &
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB          -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vsub-wasm-u8.c &

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD         -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vaddc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD         -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vaddc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD         -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vaddc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD         -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vaddc-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV         -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vdivc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV         -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vdivc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV         -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vdivc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV         -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vdivc-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX         -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmaxc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX         -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmaxc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX         -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmaxc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX         -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmaxc-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN         -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vminc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN         -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vminc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN         -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vminc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN         -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vminc-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL         -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmulc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL         -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmulc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL         -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmulc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL         -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vmulc-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=PRELU       -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vpreluc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=PRELU       -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vpreluc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=PRELU       -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vpreluc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=PRELU       -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vpreluc-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV        -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrdivc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV        -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrdivc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV        -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrdivc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV        -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrdivc-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RPRELU      -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrpreluc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RPRELU      -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrpreluc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RPRELU      -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrpreluc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RPRELU      -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrpreluc-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB        -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrsubc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB        -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrsubc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB        -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrsubc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB        -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vrsubc-wasm-u8.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB         -D BATCH_TILE=1 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vsubc-wasm-u1.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB         -D BATCH_TILE=2 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vsubc-wasm-u2.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB         -D BATCH_TILE=4 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vsubc-wasm-u4.c &
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB         -D BATCH_TILE=8 -D WASM=1 -D -o src/f32-vbinary/gen/f32-vsubc-wasm-u8.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD        -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vadd-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD        -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vadd-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=ADD        -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vadd-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV        -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV        -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=DIV        -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vdiv-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX        -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX        -D BATCH_TILE=16 -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX        -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX        -D BATCH_TILE=4  -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX        -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MAX        -D BATCH_TILE=8  -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vmax-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN        -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN        -D BATCH_TILE=16 -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN        -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN        -D BATCH_TILE=4  -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN        -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MIN        -D BATCH_TILE=8  -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vmin-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL        -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmul-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL        -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmul-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=MUL        -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmul-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=PRELU      -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vprelu-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=PRELU      -D BATCH_TILE=16 -D RELAXED=1 -D X86=0 -o src/f32-vbinary/gen/f32-vprelu-wasmrelaxedsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=PRELU      -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vprelu-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=PRELU      -D BATCH_TILE=4  -D RELAXED=1 -D X86=0 -o src/f32-vbinary/gen/f32-vprelu-wasmrelaxedsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=PRELU      -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vprelu-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=PRELU      -D BATCH_TILE=8  -D RELAXED=1 -D X86=0 -o src/f32-vbinary/gen/f32-vprelu-wasmrelaxedsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SQRDIFF    -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiff-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SQRDIFF    -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiff-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SQRDIFF    -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiff-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB        -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsub-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB        -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsub-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vop-wasmsimd.c.in -D OP=SUB        -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsub-wasmsimd-u8.c &

tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD       -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD       -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=ADD       -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vaddc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV       -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV       -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=DIV       -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vdivc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX       -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX       -D BATCH_TILE=16 -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX       -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX       -D BATCH_TILE=4  -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX       -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MAX       -D BATCH_TILE=8  -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vmaxc-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN       -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-arm-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN       -D BATCH_TILE=16 -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-x86-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN       -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-arm-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN       -D BATCH_TILE=4  -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-x86-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN       -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-arm-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MIN       -D BATCH_TILE=8  -D RELAXED=0 -D X86=1 -o src/f32-vbinary/gen/f32-vminc-wasmsimd-x86-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL       -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL       -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=MUL       -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vmulc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=PRELU     -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vpreluc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=PRELU     -D BATCH_TILE=16 -D RELAXED=1 -D X86=0 -o src/f32-vbinary/gen/f32-vpreluc-wasmrelaxedsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=PRELU     -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vpreluc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=PRELU     -D BATCH_TILE=4  -D RELAXED=1 -D X86=0 -o src/f32-vbinary/gen/f32-vpreluc-wasmrelaxedsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=PRELU     -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vpreluc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=PRELU     -D BATCH_TILE=8  -D RELAXED=1 -D X86=0 -o src/f32-vbinary/gen/f32-vpreluc-wasmrelaxedsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV      -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV      -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RDIV      -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vrdivc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RPRELU    -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vrpreluc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RPRELU    -D BATCH_TILE=16 -D RELAXED=1 -D X86=0 -o src/f32-vbinary/gen/f32-vrpreluc-wasmrelaxedsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RPRELU    -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vrpreluc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RPRELU    -D BATCH_TILE=4  -D RELAXED=1 -D X86=0 -o src/f32-vbinary/gen/f32-vrpreluc-wasmrelaxedsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RPRELU    -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vrpreluc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RPRELU    -D BATCH_TILE=8  -D RELAXED=1 -D X86=0 -o src/f32-vbinary/gen/f32-vrpreluc-wasmrelaxedsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB      -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB      -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=RSUB      -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vrsubc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SQRDIFF   -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiffc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SQRDIFF   -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiffc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SQRDIFF   -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsqrdiffc-wasmsimd-u8.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB       -D BATCH_TILE=16 -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-wasmsimd-u16.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB       -D BATCH_TILE=4  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-wasmsimd-u4.c &
tools/xngen src/f32-vbinary/vopc-wasmsimd.c.in -D OP=SUB       -D BATCH_TILE=8  -D RELAXED=0 -D X86=0 -o src/f32-vbinary/gen/f32-vsubc-wasmsimd-u8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=ADD            -D BATCH_TILE=4  -D -o src/f32-vbinary/gen/f32-vadd-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=ADD            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vadd-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=DIV            -D BATCH_TILE=4  -D -o src/f32-vbinary/gen/f32-vdiv-aarch64-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=DIV            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vdiv-aarch64-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MAX            -D BATCH_TILE=4  -D -o src/f32-vbinary/gen/f32-vmax-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MAX            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vmax-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MIN            -D BATCH_TILE=4  -D -o src/f32-vbinary/gen/f32-vmin-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MIN            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vmin-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL            -D BATCH_TILE=4  -D -o src/f32-vbinary/gen/f32-vmul-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vmul-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=PRELU          -D BATCH_TILE=4  -D -o src/f32-vbinary/gen/f32-vprelu-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=PRELU          -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vprelu-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SQRDIFF        -D BATCH_TILE=4  -D -o src/f32-vbinary/gen/f32-vsqrdiff-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SQRDIFF        -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vsqrdiff-neon-u8.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB            -D BATCH_TILE=4  -D -o src/f32-vbinary/gen/f32-vsub-neon-u4.c &
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vsub-neon-u8.c &

tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD           -D BATCH_TILE=4 -D -o src/f32-vbinary/gen/f32-vaddc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD           -D BATCH_TILE=8 -D -o src/f32-vbinary/gen/f32-vaddc-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=DIV           -D BATCH_TILE=4 -D -o src/f32-vbinary/gen/f32-vdivc-aarch64-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=DIV           -D BATCH_TILE=8 -D -o src/f32-vbinary/gen/f32-vdivc-aarch64-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MAX           -D BATCH_TILE=4 -D -o src/f32-vbinary/gen/f32-vmaxc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MAX           -D BATCH_TILE=8 -D -o src/f32-vbinary/gen/f32-vmaxc-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MIN           -D BATCH_TILE=4 -D -o src/f32-vbinary/gen/f32-vminc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MIN           -D BATCH_TILE=8 -D -o src/f32-vbinary/gen/f32-vminc-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL           -D BATCH_TILE=4 -D -o src/f32-vbinary/gen/f32-vmulc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL           -D BATCH_TILE=8 -D -o src/f32-vbinary/gen/f32-vmulc-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=PRELU         -D BATCH_TILE=4 -D -o src/f32-vbinary/gen/f32-vpreluc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=PRELU         -D BATCH_TILE=8 -D -o src/f32-vbinary/gen/f32-vpreluc-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RDIV          -D BATCH_TILE=4 -D -o src/f32-vbinary/gen/f32-vrdivc-aarch64-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RDIV          -D BATCH_TILE=8 -D -o src/f32-vbinary/gen/f32-vrdivc-aarch64-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RPRELU        -D BATCH_TILE=4 -D -o src/f32-vbinary/gen/f32-vrpreluc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RPRELU        -D BATCH_TILE=8 -D -o src/f32-vbinary/gen/f32-vrpreluc-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB          -D BATCH_TILE=4 -D -o src/f32-vbinary/gen/f32-vrsubc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB          -D BATCH_TILE=8 -D -o src/f32-vbinary/gen/f32-vrsubc-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SQRDIFF       -D BATCH_TILE=4 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SQRDIFF       -D BATCH_TILE=8 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-neon-u8.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB           -D BATCH_TILE=4 -D -o src/f32-vbinary/gen/f32-vsubc-neon-u4.c &
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB           -D BATCH_TILE=8 -D -o src/f32-vbinary/gen/f32-vsubc-neon-u8.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD             -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vadd-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD             -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vadd-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=DIV             -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vdiv-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=DIV             -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vdiv-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MAX             -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vmax-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MAX             -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vmax-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MIN             -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vmin-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MIN             -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vmin-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL             -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vmul-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL             -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vmul-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=PRELU           -D BATCH_TILE=4 -D SSE=2 -o src/f32-vbinary/gen/f32-vprelu-sse2-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=PRELU           -D BATCH_TILE=4 -D SSE=4 -o src/f32-vbinary/gen/f32-vprelu-sse41-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=PRELU           -D BATCH_TILE=8 -D SSE=2 -o src/f32-vbinary/gen/f32-vprelu-sse2-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=PRELU           -D BATCH_TILE=8 -D SSE=4 -o src/f32-vbinary/gen/f32-vprelu-sse41-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SQRDIFF         -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vsqrdiff-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SQRDIFF         -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vsqrdiff-sse-u8.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB             -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vsub-sse-u4.c &
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB             -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vsub-sse-u8.c &

tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD            -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vaddc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD            -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vaddc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=DIV            -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vdivc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=DIV            -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vdivc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MAX            -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vmaxc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MAX            -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vmaxc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MIN            -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vminc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MIN            -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vminc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL            -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vmulc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL            -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vmulc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RDIV           -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vrdivc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RDIV           -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vrdivc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB           -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vrsubc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB           -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vrsubc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SQRDIFF        -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vsqrdiffc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SQRDIFF        -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vsqrdiffc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB            -D BATCH_TILE=4 -D SSE=1 -o src/f32-vbinary/gen/f32-vsubc-sse-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB            -D BATCH_TILE=8 -D SSE=1 -o src/f32-vbinary/gen/f32-vsubc-sse-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=PRELU          -D BATCH_TILE=4 -D SSE=2 -o src/f32-vbinary/gen/f32-vpreluc-sse2-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=PRELU          -D BATCH_TILE=8 -D SSE=2 -o src/f32-vbinary/gen/f32-vpreluc-sse2-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=PRELU          -D BATCH_TILE=4 -D SSE=4 -o src/f32-vbinary/gen/f32-vpreluc-sse41-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=PRELU          -D BATCH_TILE=8 -D SSE=4 -o src/f32-vbinary/gen/f32-vpreluc-sse41-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RPRELU         -D BATCH_TILE=4 -D SSE=2 -o src/f32-vbinary/gen/f32-vrpreluc-sse2-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RPRELU         -D BATCH_TILE=8 -D SSE=2 -o src/f32-vbinary/gen/f32-vrpreluc-sse2-u8.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RPRELU         -D BATCH_TILE=4 -D SSE=4 -o src/f32-vbinary/gen/f32-vrpreluc-sse41-u4.c &
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RPRELU         -D BATCH_TILE=8 -D SSE=4 -o src/f32-vbinary/gen/f32-vrpreluc-sse41-u8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=ADD             -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vadd-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=ADD             -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vadd-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=DIV             -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vdiv-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=DIV             -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vdiv-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MAX             -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vmax-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MAX             -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vmax-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MIN             -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vmin-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MIN             -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vmin-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MUL             -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vmul-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MUL             -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vmul-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SQRDIFF         -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vsqrdiff-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SQRDIFF         -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vsqrdiff-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SUB             -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vsub-avx-u16.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SUB             -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vsub-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=PRELU           -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vprelu-avx-u8.c &
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=PRELU           -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vprelu-avx-u16.c &

tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=ADD            -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vaddc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=ADD            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vaddc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=DIV            -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vdivc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=DIV            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vdivc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MAX            -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vmaxc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MAX            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vmaxc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MIN            -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vminc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MIN            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vminc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MUL            -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vmulc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MUL            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vmulc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RDIV           -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vrdivc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RDIV           -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vrdivc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RSUB           -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vrsubc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RSUB           -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vrsubc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SQRDIFF        -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SQRDIFF        -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vsqrdiffc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SUB            -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vsubc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SUB            -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vsubc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=PRELU          -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vpreluc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=PRELU          -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vpreluc-avx-u16.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RPRELU         -D BATCH_TILE=8  -D -o src/f32-vbinary/gen/f32-vrpreluc-avx-u8.c &
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RPRELU         -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vrpreluc-avx-u16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=ADD         -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vadd-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=ADD         -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vadd-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=DIV         -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vdiv-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=DIV         -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vdiv-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MAX         -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vmax-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MAX         -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vmax-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MIN         -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vmin-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MIN         -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vmin-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MUL         -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vmul-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MUL         -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vmul-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SQRDIFF     -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vsqrdiff-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SQRDIFF     -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vsqrdiff-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SUB         -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vsub-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SUB         -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vsub-avx512f-u32.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=PRELU       -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vprelu-avx512f-u16.c &
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=PRELU       -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vprelu-avx512f-u32.c &

tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=ADD        -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vaddc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=ADD        -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vaddc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=DIV        -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vdivc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=DIV        -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vdivc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MAX        -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vmaxc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MAX        -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vmaxc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MIN        -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vminc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MIN        -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vminc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MUL        -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vmulc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MUL        -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vmulc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RDIV       -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vrdivc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RDIV       -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vrdivc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RSUB       -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vrsubc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RSUB       -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vrsubc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SQRDIFF    -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SQRDIFF    -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SUB        -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vsubc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SUB        -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vsubc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=PRELU      -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vpreluc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=PRELU      -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vpreluc-avx512f-u32.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RPRELU     -D BATCH_TILE=16 -D -o src/f32-vbinary/gen/f32-vrpreluc-avx512f-u16.c &
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RPRELU     -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vrpreluc-avx512f-u32.c &

################################### HEXAGON HVX ##################################
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=ADD             -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vadd-hvx-u128.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=ADD             -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vadd-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=ADD             -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vadd-hvx-u64.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MAX             -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vmax-hvx-u128.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MAX             -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vmax-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MAX             -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vmax-hvx-u64.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MIN             -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vmin-hvx-u128.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MIN             -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vmin-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MIN             -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vmin-hvx-u64.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MUL             -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vmul-hvx-u128.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MUL             -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vmul-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=MUL             -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vmul-hvx-u64.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SQRDIFF         -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vsqrdiff-hvx-u128.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SQRDIFF         -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vsqrdiff-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SQRDIFF         -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vsqrdiff-hvx-u64.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SUB             -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vsub-hvx-u128.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SUB             -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vsub-hvx-u32.c &
tools/xngen src/f32-vbinary/vop-hvx.c.in -D OP=SUB             -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vsub-hvx-u64.c &

tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=ADD            -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vaddc-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=ADD            -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vaddc-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=ADD            -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vaddc-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MAX            -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vmaxc-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MAX            -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vmaxc-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MAX            -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vmaxc-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MIN            -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vminc-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MIN            -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vminc-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MIN            -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vminc-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MUL            -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vmulc-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MUL            -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vmulc-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=MUL            -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vmulc-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=RSUB           -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vrsubc-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=RSUB           -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vrsubc-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=RSUB           -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vrsubc-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SQRDIFF        -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SQRDIFF        -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SQRDIFF        -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-hvx-u64.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SUB            -D BATCH_TILE=128 -D -o src/f32-vbinary/gen/f32-vsubc-hvx-u128.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SUB            -D BATCH_TILE=32 -D -o src/f32-vbinary/gen/f32-vsubc-hvx-u32.c &
tools/xngen src/f32-vbinary/vopc-hvx.c.in -D OP=SUB            -D BATCH_TILE=64 -D -o src/f32-vbinary/gen/f32-vsubc-hvx-u64.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=ADD       -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vadd-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=ADD       -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vadd-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=DIV       -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vdiv-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=DIV       -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vdiv-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MAX       -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vmax-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MAX       -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vmax-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MIN       -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vmin-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MIN       -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vmin-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MUL       -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vmul-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=MUL       -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vmul-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=SQRDIFF   -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vsqrdiff-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=SQRDIFF   -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vsqrdiff-rvv-u8v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=SUB       -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vsub-rvv-u4v.c &
tools/xngen src/f32-vbinary/vop-rvv.c.in -D OP=SUB       -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vsub-rvv-u8v.c &

tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=ADD      -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vaddc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=ADD      -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vaddc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=DIV      -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vdivc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=DIV      -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vdivc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MAX      -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vmaxc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MAX      -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vmaxc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MIN      -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vminc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MIN      -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vminc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MUL      -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vmulc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=MUL      -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vmulc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=RDIV     -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vrdivc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=RDIV     -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vrdivc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=RSUB     -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vrsubc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=RSUB     -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vrsubc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=SQRDIFF  -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=SQRDIFF  -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vsqrdiffc-rvv-u8v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=SUB      -D LMUL=4 -D -o src/f32-vbinary/gen/f32-vsubc-rvv-u4v.c &
tools/xngen src/f32-vbinary/vopc-rvv.c.in -D OP=SUB      -D LMUL=8 -D -o src/f32-vbinary/gen/f32-vsubc-rvv-u8v.c &

wait
