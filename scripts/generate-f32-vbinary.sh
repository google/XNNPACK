#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vadd-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vadd-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vadd-scalar-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vdiv-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vdiv-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vdiv-scalar-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vmax-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vmax-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vmax-scalar-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vmin-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vmin-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vmin-scalar-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vmul-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vmul-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vmul-scalar-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vsub-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vsub-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vsub-scalar-x4.c

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vaddc-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vaddc-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vaddc-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vdivc-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vdivc-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vdivc-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vrdivc-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vrdivc-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vrdivc-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vmaxc-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vmaxc-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vmaxc-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vminc-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vminc-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vminc-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vmulc-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vmulc-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vmulc-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vsubc-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vsubc-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vsubc-scalar-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vrsubc-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vrsubc-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vrsubc-scalar-x4.c

### WAsm-specific micro-kernels
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vadd-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vadd-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vadd-wasm-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vdiv-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vdiv-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=DIV -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vdiv-wasm-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vmax-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vmax-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MAX -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vmax-wasm-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vmin-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vmin-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MIN -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vmin-wasm-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vmul-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vmul-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vmul-wasm-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vsub-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vsub-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vsub-wasm-x4.c

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vaddc-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vaddc-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vaddc-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vdivc-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vdivc-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=DIV  -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vdivc-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vrdivc-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vrdivc-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RDIV -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vrdivc-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vmaxc-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vmaxc-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MAX  -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vmaxc-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vminc-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vminc-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MIN  -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vminc-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vmulc-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vmulc-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=MUL  -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vmulc-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vsubc-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vsubc-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=SUB  -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vsubc-wasm-x4.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vrsubc-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vrsubc-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vrsubc-wasm-x4.c

################################### ARM NEON ##################################
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=ADD -D BATCH_TILE=4 -o src/f32-vbinary/gen/vadd-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=ADD -D BATCH_TILE=8 -o src/f32-vbinary/gen/vadd-neon-x8.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=DIV -D BATCH_TILE=4 -o src/f32-vbinary/gen/vdiv-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=DIV -D BATCH_TILE=8 -o src/f32-vbinary/gen/vdiv-neon-x8.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MAX -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmax-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MAX -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmax-neon-x8.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MIN -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmin-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MIN -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmin-neon-x8.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmul-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmul-neon-x8.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsub-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsub-neon-x8.c

tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vaddc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vaddc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=DIV  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vdivc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=DIV  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vdivc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RDIV -D BATCH_TILE=4 -o src/f32-vbinary/gen/vrdivc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RDIV -D BATCH_TILE=8 -o src/f32-vbinary/gen/vrdivc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MAX  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmaxc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MAX  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmaxc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MIN  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vminc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MIN  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vminc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmulc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmulc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsubc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsubc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vrsubc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vrsubc-neon-x8.c

#################################### PSIMD ####################################
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=ADD -D BATCH_TILE=4 -o src/f32-vbinary/gen/vadd-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=ADD -D BATCH_TILE=8 -o src/f32-vbinary/gen/vadd-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=DIV -D BATCH_TILE=4 -o src/f32-vbinary/gen/vdiv-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=DIV -D BATCH_TILE=8 -o src/f32-vbinary/gen/vdiv-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MAX -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmax-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MAX -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmax-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MIN -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmin-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MIN -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmin-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MUL -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmul-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MUL -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmul-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=SUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsub-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=SUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsub-psimd-x8.c

tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=ADD  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vaddc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=ADD  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vaddc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=DIV  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vdivc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=DIV  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vdivc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=RDIV -D BATCH_TILE=4 -o src/f32-vbinary/gen/vrdivc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=RDIV -D BATCH_TILE=8 -o src/f32-vbinary/gen/vrdivc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MAX  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmaxc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MAX  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmaxc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MIN  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vminc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MIN  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vminc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MUL  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmulc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MUL  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmulc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=SUB  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsubc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=SUB  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsubc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=RSUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vrsubc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=RSUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vrsubc-psimd-x8.c

################################# x86 128-bit #################################
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD -D BATCH_TILE=4 -o src/f32-vbinary/gen/vadd-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD -D BATCH_TILE=8 -o src/f32-vbinary/gen/vadd-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=DIV -D BATCH_TILE=4 -o src/f32-vbinary/gen/vdiv-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=DIV -D BATCH_TILE=8 -o src/f32-vbinary/gen/vdiv-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MAX -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmax-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MAX -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmax-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MIN -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmin-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MIN -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmin-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmul-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmul-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsub-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsub-sse-x8.c

tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vaddc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vaddc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=DIV  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vdivc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=DIV  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vdivc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RDIV -D BATCH_TILE=4 -o src/f32-vbinary/gen/vrdivc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RDIV -D BATCH_TILE=8 -o src/f32-vbinary/gen/vrdivc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MAX  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmaxc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MAX  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmaxc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MIN  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vminc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MIN  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vminc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmulc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmulc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsubc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsubc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vrsubc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vrsubc-sse-x8.c

################################# x86 256-bit #################################
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=ADD -D BATCH_TILE=8  -o src/f32-vbinary/gen/vadd-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=ADD -D BATCH_TILE=16 -o src/f32-vbinary/gen/vadd-avx-x16.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=DIV -D BATCH_TILE=8  -o src/f32-vbinary/gen/vdiv-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=DIV -D BATCH_TILE=16 -o src/f32-vbinary/gen/vdiv-avx-x16.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MAX -D BATCH_TILE=8  -o src/f32-vbinary/gen/vmax-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MAX -D BATCH_TILE=16 -o src/f32-vbinary/gen/vmax-avx-x16.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MIN -D BATCH_TILE=8  -o src/f32-vbinary/gen/vmin-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MIN -D BATCH_TILE=16 -o src/f32-vbinary/gen/vmin-avx-x16.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MUL -D BATCH_TILE=8  -o src/f32-vbinary/gen/vmul-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=MUL -D BATCH_TILE=16 -o src/f32-vbinary/gen/vmul-avx-x16.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SUB -D BATCH_TILE=8  -o src/f32-vbinary/gen/vsub-avx-x8.c
tools/xngen src/f32-vbinary/vop-avx.c.in -D OP=SUB -D BATCH_TILE=16 -o src/f32-vbinary/gen/vsub-avx-x16.c

tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=ADD  -D BATCH_TILE=8  -o src/f32-vbinary/gen/vaddc-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=ADD  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vaddc-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=DIV  -D BATCH_TILE=8  -o src/f32-vbinary/gen/vdivc-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=DIV  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vdivc-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RDIV -D BATCH_TILE=8  -o src/f32-vbinary/gen/vrdivc-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RDIV -D BATCH_TILE=16 -o src/f32-vbinary/gen/vrdivc-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MAX  -D BATCH_TILE=8  -o src/f32-vbinary/gen/vmaxc-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MAX  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vmaxc-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MIN  -D BATCH_TILE=8  -o src/f32-vbinary/gen/vminc-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MIN  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vminc-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MUL  -D BATCH_TILE=8  -o src/f32-vbinary/gen/vmulc-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=MUL  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vmulc-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SUB  -D BATCH_TILE=8  -o src/f32-vbinary/gen/vsubc-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=SUB  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vsubc-avx-x16.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RSUB -D BATCH_TILE=8  -o src/f32-vbinary/gen/vrsubc-avx-x8.c
tools/xngen src/f32-vbinary/vopc-avx.c.in -D OP=RSUB -D BATCH_TILE=16 -o src/f32-vbinary/gen/vrsubc-avx-x16.c

################################# x86 512-bit #################################
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=ADD -D BATCH_TILE=16 -o src/f32-vbinary/gen/vadd-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=ADD -D BATCH_TILE=32 -o src/f32-vbinary/gen/vadd-avx512f-x32.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=DIV -D BATCH_TILE=16 -o src/f32-vbinary/gen/vdiv-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=DIV -D BATCH_TILE=32 -o src/f32-vbinary/gen/vdiv-avx512f-x32.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MAX -D BATCH_TILE=16 -o src/f32-vbinary/gen/vmax-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MAX -D BATCH_TILE=32 -o src/f32-vbinary/gen/vmax-avx512f-x32.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MIN -D BATCH_TILE=16 -o src/f32-vbinary/gen/vmin-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MIN -D BATCH_TILE=32 -o src/f32-vbinary/gen/vmin-avx512f-x32.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MUL -D BATCH_TILE=16 -o src/f32-vbinary/gen/vmul-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=MUL -D BATCH_TILE=32 -o src/f32-vbinary/gen/vmul-avx512f-x32.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SUB -D BATCH_TILE=16 -o src/f32-vbinary/gen/vsub-avx512f-x16.c
tools/xngen src/f32-vbinary/vop-avx512f.c.in -D OP=SUB -D BATCH_TILE=32 -o src/f32-vbinary/gen/vsub-avx512f-x32.c

tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=ADD  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vaddc-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=ADD  -D BATCH_TILE=32 -o src/f32-vbinary/gen/vaddc-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=DIV  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vdivc-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=DIV  -D BATCH_TILE=32 -o src/f32-vbinary/gen/vdivc-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RDIV -D BATCH_TILE=16 -o src/f32-vbinary/gen/vrdivc-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RDIV -D BATCH_TILE=32 -o src/f32-vbinary/gen/vrdivc-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MAX  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vmaxc-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MAX  -D BATCH_TILE=32 -o src/f32-vbinary/gen/vmaxc-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MIN  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vminc-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MIN  -D BATCH_TILE=32 -o src/f32-vbinary/gen/vminc-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MUL  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vmulc-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=MUL  -D BATCH_TILE=32 -o src/f32-vbinary/gen/vmulc-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SUB  -D BATCH_TILE=16 -o src/f32-vbinary/gen/vsubc-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=SUB  -D BATCH_TILE=32 -o src/f32-vbinary/gen/vsubc-avx512f-x32.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RSUB -D BATCH_TILE=16 -o src/f32-vbinary/gen/vrsubc-avx512f-x16.c
tools/xngen src/f32-vbinary/vopc-avx512f.c.in -D OP=RSUB -D BATCH_TILE=32 -o src/f32-vbinary/gen/vrsubc-avx512f-x32.c

################################## Unit tests #################################
tools/generate-vbinary-test.py --spec test/f32-vadd.yaml --output test/f32-vadd.cc
tools/generate-vbinary-test.py --spec test/f32-vdiv.yaml --output test/f32-vdiv.cc
tools/generate-vbinary-test.py --spec test/f32-vmax.yaml --output test/f32-vmax.cc
tools/generate-vbinary-test.py --spec test/f32-vmin.yaml --output test/f32-vmin.cc
tools/generate-vbinary-test.py --spec test/f32-vmul.yaml --output test/f32-vmul.cc
tools/generate-vbinary-test.py --spec test/f32-vsub.yaml --output test/f32-vsub.cc
tools/generate-vbinary-test.py --spec test/f32-vaddc.yaml --output test/f32-vaddc.cc
tools/generate-vbinary-test.py --spec test/f32-vdivc.yaml --output test/f32-vdivc.cc
tools/generate-vbinary-test.py --spec test/f32-vrdivc.yaml --output test/f32-vrdivc.cc
tools/generate-vbinary-test.py --spec test/f32-vmaxc.yaml --output test/f32-vmaxc.cc
tools/generate-vbinary-test.py --spec test/f32-vminc.yaml --output test/f32-vminc.cc
tools/generate-vbinary-test.py --spec test/f32-vmulc.yaml --output test/f32-vmulc.cc
tools/generate-vbinary-test.py --spec test/f32-vsubc.yaml --output test/f32-vsubc.cc
tools/generate-vbinary-test.py --spec test/f32-vrsubc.yaml --output test/f32-vrsubc.cc
