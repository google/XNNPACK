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
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vmul-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vmul-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vmul-scalar-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vsub-scalar-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vsub-scalar-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vsub-scalar-x4.c

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=1 -D WASM=0 -o src/f32-vbinary/gen/vaddc-scalar-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=2 -D WASM=0 -o src/f32-vbinary/gen/vaddc-scalar-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=4 -D WASM=0 -o src/f32-vbinary/gen/vaddc-scalar-x4.c
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
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vmul-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vmul-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vmul-wasm-x4.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vsub-wasm-x1.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vsub-wasm-x2.c
tools/xngen src/f32-vbinary/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vsub-wasm-x4.c

tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=1 -D WASM=1 -o src/f32-vbinary/gen/vaddc-wasm-x1.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=2 -D WASM=1 -o src/f32-vbinary/gen/vaddc-wasm-x2.c
tools/xngen src/f32-vbinary/vopc-scalar.c.in -D OP=ADD  -D BATCH_TILE=4 -D WASM=1 -o src/f32-vbinary/gen/vaddc-wasm-x4.c
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
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmul-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=MUL -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmul-neon-x8.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsub-neon-x4.c
tools/xngen src/f32-vbinary/vop-neon.c.in -D OP=SUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsub-neon-x8.c

tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vaddc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=ADD  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vaddc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmulc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=MUL  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmulc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsubc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=SUB  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsubc-neon-x8.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vrsubc-neon-x4.c
tools/xngen src/f32-vbinary/vopc-neon.c.in -D OP=RSUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vrsubc-neon-x8.c

#################################### PSIMD ####################################
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=ADD -D BATCH_TILE=4 -o src/f32-vbinary/gen/vadd-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=ADD -D BATCH_TILE=8 -o src/f32-vbinary/gen/vadd-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MUL -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmul-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=MUL -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmul-psimd-x8.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=SUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsub-psimd-x4.c
tools/xngen src/f32-vbinary/vop-psimd.c.in -D OP=SUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsub-psimd-x8.c

tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=ADD  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vaddc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=ADD  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vaddc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MUL  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmulc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=MUL  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmulc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=SUB  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsubc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=SUB  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsubc-psimd-x8.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=RSUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vrsubc-psimd-x4.c
tools/xngen src/f32-vbinary/vopc-psimd.c.in -D OP=RSUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vrsubc-psimd-x8.c

################################### x86 SSE ###################################
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD -D BATCH_TILE=4 -o src/f32-vbinary/gen/vadd-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=ADD -D BATCH_TILE=8 -o src/f32-vbinary/gen/vadd-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmul-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=MUL -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmul-sse-x8.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsub-sse-x4.c
tools/xngen src/f32-vbinary/vop-sse.c.in -D OP=SUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsub-sse-x8.c

tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vaddc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=ADD  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vaddc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vmulc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=MUL  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vmulc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB  -D BATCH_TILE=4 -o src/f32-vbinary/gen/vsubc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=SUB  -D BATCH_TILE=8 -o src/f32-vbinary/gen/vsubc-sse-x8.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB -D BATCH_TILE=4 -o src/f32-vbinary/gen/vrsubc-sse-x4.c
tools/xngen src/f32-vbinary/vopc-sse.c.in -D OP=RSUB -D BATCH_TILE=8 -o src/f32-vbinary/gen/vrsubc-sse-x8.c

################################## Unit tests #################################
tools/generate-vbinary-test.py --spec test/f32-vadd.yaml --output test/f32-vadd.cc
tools/generate-vbinary-test.py --spec test/f32-vmul.yaml --output test/f32-vmul.cc
tools/generate-vbinary-test.py --spec test/f32-vsub.yaml --output test/f32-vsub.cc
tools/generate-vbinary-test.py --spec test/f32-vaddc.yaml --output test/f32-vaddc.cc
tools/generate-vbinary-test.py --spec test/f32-vmulc.yaml --output test/f32-vmulc.cc
tools/generate-vbinary-test.py --spec test/f32-vsubc.yaml --output test/f32-vsubc.cc
tools/generate-vbinary-test.py --spec test/f32-vrsubc.yaml --output test/f32-vrsubc.cc
