#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vmulcaddc/scalar.c.in -D CHANNEL_TILE=1 -D ROW_TILE=2 -D WASM=0 -o src/f32-vmulcaddc/gen/c1-scalar-2x.c
tools/xngen src/f32-vmulcaddc/scalar.c.in -D CHANNEL_TILE=2 -D ROW_TILE=2 -D WASM=0 -o src/f32-vmulcaddc/gen/c2-scalar-2x.c
tools/xngen src/f32-vmulcaddc/scalar.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D WASM=0 -o src/f32-vmulcaddc/gen/c4-scalar-2x.c

### WAsm-specific micro-kernels
tools/xngen src/f32-vmulcaddc/scalar.c.in -D CHANNEL_TILE=1 -D ROW_TILE=2 -D WASM=1 -o src/f32-vmulcaddc/gen/c1-wasm-2x.c
tools/xngen src/f32-vmulcaddc/scalar.c.in -D CHANNEL_TILE=2 -D ROW_TILE=2 -D WASM=1 -o src/f32-vmulcaddc/gen/c2-wasm-2x.c
tools/xngen src/f32-vmulcaddc/scalar.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D WASM=1 -o src/f32-vmulcaddc/gen/c4-wasm-2x.c

################################### ARM NEON ##################################
tools/xngen src/f32-vmulcaddc/neon.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D FMA=0 -o src/f32-vmulcaddc/gen/c4-neon-2x.c
tools/xngen src/f32-vmulcaddc/neon.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -D FMA=0 -o src/f32-vmulcaddc/gen/c8-neon-2x.c

tools/xngen src/f32-vmulcaddc/neon.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -D FMA=1 -o src/f32-vmulcaddc/gen/c4-neonfma-2x.c
tools/xngen src/f32-vmulcaddc/neon.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -D FMA=1 -o src/f32-vmulcaddc/gen/c8-neonfma-2x.c

#################################### PSIMD ####################################
tools/xngen src/f32-vmulcaddc/psimd.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -o src/f32-vmulcaddc/gen/c4-psimd-2x.c
tools/xngen src/f32-vmulcaddc/psimd.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -o src/f32-vmulcaddc/gen/c8-psimd-2x.c

################################### x86 SSE ###################################
tools/xngen src/f32-vmulcaddc/sse.c.in -D CHANNEL_TILE=4 -D ROW_TILE=2 -o src/f32-vmulcaddc/gen/c4-sse-2x.c
tools/xngen src/f32-vmulcaddc/sse.c.in -D CHANNEL_TILE=8 -D ROW_TILE=2 -o src/f32-vmulcaddc/gen/c8-sse-2x.c

################################## Unit tests #################################
tools/generate-vmulcaddc-test.py --spec test/f32-vmulcaddc.yaml --output test/f32-vmulcaddc.cc
