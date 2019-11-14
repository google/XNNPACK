#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-binop/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=1 -o src/f32-binop/vadd-scalar-x1.c
tools/xngen src/f32-binop/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=2 -o src/f32-binop/vadd-scalar-x2.c
tools/xngen src/f32-binop/vop-scalar.c.in -D OP=ADD -D BATCH_TILE=4 -o src/f32-binop/vadd-scalar-x4.c
tools/xngen src/f32-binop/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -o src/f32-binop/vmul-scalar-x1.c
tools/xngen src/f32-binop/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -o src/f32-binop/vmul-scalar-x2.c
tools/xngen src/f32-binop/vop-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -o src/f32-binop/vmul-scalar-x4.c
tools/xngen src/f32-binop/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -o src/f32-binop/vsub-scalar-x1.c
tools/xngen src/f32-binop/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -o src/f32-binop/vsub-scalar-x2.c
tools/xngen src/f32-binop/vop-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -o src/f32-binop/vsub-scalar-x4.c

tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=ADD -D BATCH_TILE=1 -o src/f32-binop/vaddc-scalar-x1.c
tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=ADD -D BATCH_TILE=2 -o src/f32-binop/vaddc-scalar-x2.c
tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=ADD -D BATCH_TILE=4 -o src/f32-binop/vaddc-scalar-x4.c
tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=MUL -D BATCH_TILE=1 -o src/f32-binop/vmulc-scalar-x1.c
tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=MUL -D BATCH_TILE=2 -o src/f32-binop/vmulc-scalar-x2.c
tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=MUL -D BATCH_TILE=4 -o src/f32-binop/vmulc-scalar-x4.c
tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=SUB -D BATCH_TILE=1 -o src/f32-binop/vsubc-scalar-x1.c
tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=SUB -D BATCH_TILE=2 -o src/f32-binop/vsubc-scalar-x2.c
tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=SUB -D BATCH_TILE=4 -o src/f32-binop/vsubc-scalar-x4.c
tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=1 -o src/f32-binop/vrsubc-scalar-x1.c
tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=2 -o src/f32-binop/vrsubc-scalar-x2.c
tools/xngen src/f32-binop/vopc-scalar.c.in -D OP=RSUB -D BATCH_TILE=4 -o src/f32-binop/vrsubc-scalar-x4.c

################################### ARM NEON ##################################
tools/xngen src/f32-binop/vop-neon.c.in -D OP=ADD -D BATCH_TILE=4 -o src/f32-binop/vadd-neon-x4.c
tools/xngen src/f32-binop/vop-neon.c.in -D OP=ADD -D BATCH_TILE=8 -o src/f32-binop/vadd-neon-x8.c
tools/xngen src/f32-binop/vop-neon.c.in -D OP=MUL -D BATCH_TILE=4 -o src/f32-binop/vmul-neon-x4.c
tools/xngen src/f32-binop/vop-neon.c.in -D OP=MUL -D BATCH_TILE=8 -o src/f32-binop/vmul-neon-x8.c
tools/xngen src/f32-binop/vop-neon.c.in -D OP=SUB -D BATCH_TILE=4 -o src/f32-binop/vsub-neon-x4.c
tools/xngen src/f32-binop/vop-neon.c.in -D OP=SUB -D BATCH_TILE=8 -o src/f32-binop/vsub-neon-x8.c

tools/xngen src/f32-binop/vopc-neon.c.in -D OP=ADD  -D BATCH_TILE=4 -o src/f32-binop/vaddc-neon-x4.c
tools/xngen src/f32-binop/vopc-neon.c.in -D OP=ADD  -D BATCH_TILE=8 -o src/f32-binop/vaddc-neon-x8.c
tools/xngen src/f32-binop/vopc-neon.c.in -D OP=MUL  -D BATCH_TILE=4 -o src/f32-binop/vmulc-neon-x4.c
tools/xngen src/f32-binop/vopc-neon.c.in -D OP=MUL  -D BATCH_TILE=8 -o src/f32-binop/vmulc-neon-x8.c
tools/xngen src/f32-binop/vopc-neon.c.in -D OP=SUB  -D BATCH_TILE=4 -o src/f32-binop/vsubc-neon-x4.c
tools/xngen src/f32-binop/vopc-neon.c.in -D OP=SUB  -D BATCH_TILE=8 -o src/f32-binop/vsubc-neon-x8.c
tools/xngen src/f32-binop/vopc-neon.c.in -D OP=RSUB -D BATCH_TILE=4 -o src/f32-binop/vrsubc-neon-x4.c
tools/xngen src/f32-binop/vopc-neon.c.in -D OP=RSUB -D BATCH_TILE=8 -o src/f32-binop/vrsubc-neon-x8.c

#################################### PSIMD ####################################
tools/xngen src/f32-binop/vop-psimd.c.in -D OP=ADD -D BATCH_TILE=4 -o src/f32-binop/vadd-psimd-x4.c
tools/xngen src/f32-binop/vop-psimd.c.in -D OP=ADD -D BATCH_TILE=8 -o src/f32-binop/vadd-psimd-x8.c
tools/xngen src/f32-binop/vop-psimd.c.in -D OP=MUL -D BATCH_TILE=4 -o src/f32-binop/vmul-psimd-x4.c
tools/xngen src/f32-binop/vop-psimd.c.in -D OP=MUL -D BATCH_TILE=8 -o src/f32-binop/vmul-psimd-x8.c
tools/xngen src/f32-binop/vop-psimd.c.in -D OP=SUB -D BATCH_TILE=4 -o src/f32-binop/vsub-psimd-x4.c
tools/xngen src/f32-binop/vop-psimd.c.in -D OP=SUB -D BATCH_TILE=8 -o src/f32-binop/vsub-psimd-x8.c

tools/xngen src/f32-binop/vopc-psimd.c.in -D OP=ADD  -D BATCH_TILE=4 -o src/f32-binop/vaddc-psimd-x4.c
tools/xngen src/f32-binop/vopc-psimd.c.in -D OP=ADD  -D BATCH_TILE=8 -o src/f32-binop/vaddc-psimd-x8.c
tools/xngen src/f32-binop/vopc-psimd.c.in -D OP=MUL  -D BATCH_TILE=4 -o src/f32-binop/vmulc-psimd-x4.c
tools/xngen src/f32-binop/vopc-psimd.c.in -D OP=MUL  -D BATCH_TILE=8 -o src/f32-binop/vmulc-psimd-x8.c
tools/xngen src/f32-binop/vopc-psimd.c.in -D OP=SUB  -D BATCH_TILE=4 -o src/f32-binop/vsubc-psimd-x4.c
tools/xngen src/f32-binop/vopc-psimd.c.in -D OP=SUB  -D BATCH_TILE=8 -o src/f32-binop/vsubc-psimd-x8.c
tools/xngen src/f32-binop/vopc-psimd.c.in -D OP=RSUB -D BATCH_TILE=4 -o src/f32-binop/vrsubc-psimd-x4.c
tools/xngen src/f32-binop/vopc-psimd.c.in -D OP=RSUB -D BATCH_TILE=8 -o src/f32-binop/vrsubc-psimd-x8.c

################################### x86 SSE ###################################
tools/xngen src/f32-binop/vop-sse.c.in -D OP=ADD -D BATCH_TILE=4 -o src/f32-binop/vadd-sse-x4.c
tools/xngen src/f32-binop/vop-sse.c.in -D OP=ADD -D BATCH_TILE=8 -o src/f32-binop/vadd-sse-x8.c
tools/xngen src/f32-binop/vop-sse.c.in -D OP=MUL -D BATCH_TILE=4 -o src/f32-binop/vmul-sse-x4.c
tools/xngen src/f32-binop/vop-sse.c.in -D OP=MUL -D BATCH_TILE=8 -o src/f32-binop/vmul-sse-x8.c
tools/xngen src/f32-binop/vop-sse.c.in -D OP=SUB -D BATCH_TILE=4 -o src/f32-binop/vsub-sse-x4.c
tools/xngen src/f32-binop/vop-sse.c.in -D OP=SUB -D BATCH_TILE=8 -o src/f32-binop/vsub-sse-x8.c

tools/xngen src/f32-binop/vopc-sse.c.in -D OP=ADD  -D BATCH_TILE=4 -o src/f32-binop/vaddc-sse-x4.c
tools/xngen src/f32-binop/vopc-sse.c.in -D OP=ADD  -D BATCH_TILE=8 -o src/f32-binop/vaddc-sse-x8.c
tools/xngen src/f32-binop/vopc-sse.c.in -D OP=MUL  -D BATCH_TILE=4 -o src/f32-binop/vmulc-sse-x4.c
tools/xngen src/f32-binop/vopc-sse.c.in -D OP=MUL  -D BATCH_TILE=8 -o src/f32-binop/vmulc-sse-x8.c
tools/xngen src/f32-binop/vopc-sse.c.in -D OP=SUB  -D BATCH_TILE=4 -o src/f32-binop/vsubc-sse-x4.c
tools/xngen src/f32-binop/vopc-sse.c.in -D OP=SUB  -D BATCH_TILE=8 -o src/f32-binop/vsubc-sse-x8.c
tools/xngen src/f32-binop/vopc-sse.c.in -D OP=RSUB -D BATCH_TILE=4 -o src/f32-binop/vrsubc-sse-x4.c
tools/xngen src/f32-binop/vopc-sse.c.in -D OP=RSUB -D BATCH_TILE=8 -o src/f32-binop/vrsubc-sse-x8.c

################################## Unit tests #################################
tools/generate-vbinop-test.py --spec test/f32-vadd.yaml --output test/f32-vadd.cc
tools/generate-vbinop-test.py --spec test/f32-vmul.yaml --output test/f32-vmul.cc
tools/generate-vbinop-test.py --spec test/f32-vsub.yaml --output test/f32-vsub.cc
tools/generate-vbinop-test.py --spec test/f32-vaddc.yaml --output test/f32-vaddc.cc
tools/generate-vbinop-test.py --spec test/f32-vmulc.yaml --output test/f32-vmulc.cc
tools/generate-vbinop-test.py --spec test/f32-vsubc.yaml --output test/f32-vsubc.cc
tools/generate-vbinop-test.py --spec test/f32-vrsubc.yaml --output test/f32-vrsubc.cc
