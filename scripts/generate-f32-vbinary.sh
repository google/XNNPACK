#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-scalar-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-scalar-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-scalar-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-scalar-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-scalar-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-scalar-x4.c

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=4 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-scalar-x4.c

### WAsm-specific micro-kernels
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-wasm-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-wasm-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-wasm-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-wasm-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-wasm-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-wasm-x4.c

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=4 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-wasm-x4.c

################################### ARM NEON ##################################
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=ADD -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=ADD -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-neon-x8.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=DIV -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=DIV -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-neon-x8.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MAX -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MAX -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-neon-x8.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MIN -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MIN -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-neon-x8.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-neon-x8.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-neon-x8.c

tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=DIV  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=DIV  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RDIV -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RDIV -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MAX  -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MAX  -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MIN  -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MIN  -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-neon-x8.c

#################################### PSIMD ####################################
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=ADD -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=ADD -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=DIV -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=DIV -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MAX -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MAX -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MIN -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MIN -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MUL -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MUL -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=SUB -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=SUB -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-psimd-x8.c

tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=ADD  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=ADD  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=DIV  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=DIV  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=RDIV -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=RDIV -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MAX  -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MAX  -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MIN  -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MIN  -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MUL  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MUL  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=SUB  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=SUB  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=RSUB -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=RSUB -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-psimd-x8.c

################################# x86 128-bit #################################
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=DIV -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=DIV -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MAX -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MAX -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MIN -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MIN -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-sse-x8.c

tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=DIV  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=DIV  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RDIV -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RDIV -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MAX  -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MAX  -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MIN  -D BATCH_TILE=4 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MIN  -D BATCH_TILE=8 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB  -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB  -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB -D BATCH_TILE=4 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB -D BATCH_TILE=8 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-sse-x8.c

################################# x86 256-bit #################################
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=ADD -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=ADD -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-avx-x16.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=DIV -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=DIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-avx-x16.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MAX -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MAX -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-avx-x16.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MIN -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MIN -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-avx-x16.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MUL -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MUL -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-avx-x16.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SUB -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-avx-x16.c

tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=ADD  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=ADD  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=DIV  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=DIV  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RDIV -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RDIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MAX  -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MAX  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MIN  -D BATCH_TILE=8  -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MIN  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MUL  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MUL  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SUB  -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SUB  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RSUB -D BATCH_TILE=8  -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RSUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-avx-x16.c

################################# x86 512-bit #################################
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=ADD -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=ADD -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vadd-minmax-avx512f-x32.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=DIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=DIV -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdiv-minmax-avx512f-x32.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MAX -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MAX -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmax-avx512f-x32.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MIN -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MIN -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmin-avx512f-x32.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MUL -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MUL -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmul-minmax-avx512f-x32.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SUB -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsub-minmax-avx512f-x32.c

tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=ADD  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=ADD  -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vaddc-minmax-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=DIV  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=DIV  -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vdivc-minmax-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RDIV -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RDIV -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrdivc-minmax-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MAX  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MAX  -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vmaxc-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MIN  -D BATCH_TILE=16 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MIN  -D BATCH_TILE=32 -D ACTIVATION=LINEAR -o src/f32-vbinary/gen/vminc-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MUL  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MUL  -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vmulc-minmax-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SUB  -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SUB  -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vsubc-minmax-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RSUB -D BATCH_TILE=16 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RSUB -D BATCH_TILE=32 -D ACTIVATION=MINMAX -o src/f32-vbinary/gen/vrsubc-minmax-avx512f-x32.c

################################## Unit tests #################################
tools/generate-vbinary-test.py --spec test/f32-vadd-minmax.yaml --output test/f32-vadd-minmax.cc
tools/generate-vbinary-test.py --spec test/f32-vdiv-minmax.yaml --output test/f32-vdiv-minmax.cc
tools/generate-vbinary-test.py --spec test/f32-vmax.yaml --output test/f32-vmax.cc
tools/generate-vbinary-test.py --spec test/f32-vmin.yaml --output test/f32-vmin.cc
tools/generate-vbinary-test.py --spec test/f32-vmul-minmax.yaml --output test/f32-vmul-minmax.cc
tools/generate-vbinary-test.py --spec test/f32-vsub-minmax.yaml --output test/f32-vsub-minmax.cc
tools/generate-vbinary-test.py --spec test/f32-vaddc-minmax.yaml --output test/f32-vaddc-minmax.cc
tools/generate-vbinary-test.py --spec test/f32-vdivc-minmax.yaml --output test/f32-vdivc-minmax.cc
tools/generate-vbinary-test.py --spec test/f32-vrdivc-minmax.yaml --output test/f32-vrdivc-minmax.cc
tools/generate-vbinary-test.py --spec test/f32-vmaxc.yaml --output test/f32-vmaxc.cc
tools/generate-vbinary-test.py --spec test/f32-vminc.yaml --output test/f32-vminc.cc
tools/generate-vbinary-test.py --spec test/f32-vmulc-minmax.yaml --output test/f32-vmulc-minmax.cc
tools/generate-vbinary-test.py --spec test/f32-vsubc-minmax.yaml --output test/f32-vsubc-minmax.cc
tools/generate-vbinary-test.py --spec test/f32-vrsubc-minmax.yaml --output test/f32-vrsubc-minmax.cc
