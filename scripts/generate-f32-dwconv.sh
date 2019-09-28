#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-dwconv/up-scalar.c.in -D CR=1 -D MR=4  -D AR=2 -o src/f32-dwconv/up1x4-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CR=1 -D MR=9  -D AR=2 -o src/f32-dwconv/up1x9-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CR=1 -D MR=25 -D AR=2 -o src/f32-dwconv/up1x25-scalar.c

################################### ARM NEON ##################################
tools/xngen src/f32-dwconv/up-neon.c.in -D CR=4 -D MR=9 -D AR=1 -D FMA=0 -o src/f32-dwconv/up4x9-neon.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CR=4 -D MR=9 -D AR=1 -D FMA=1 -o src/f32-dwconv/up4x9-neonfma.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CR=8 -D MR=9 -D AR=1 -D FMA=1 -o src/f32-dwconv/up8x9-neonfma.c

#################################### PSIMD ####################################
tools/xngen src/f32-dwconv/up-psimd.c.in -D CR=4 -D MR=4 -D AR=2 -o src/f32-dwconv/up4x4-psimd.c
tools/xngen src/f32-dwconv/up-psimd.c.in -D CR=4 -D MR=9 -D AR=2 -o src/f32-dwconv/up4x9-psimd.c
tools/xngen src/f32-dwconv/up-psimd.c.in -D CR=4 -D MR=25 -D AR=2 -o src/f32-dwconv/up4x25-psimd.c

################################### x86 SSE ###################################
tools/xngen src/f32-dwconv/up-sse.c.in -D CR=4 -D MR=4 -D AR=2 -o src/f32-dwconv/up4x4-sse.c
tools/xngen src/f32-dwconv/up-sse.c.in -D CR=4 -D MR=9 -D AR=2 -o src/f32-dwconv/up4x9-sse.c
tools/xngen src/f32-dwconv/up-sse.c.in -D CR=4 -D MR=25 -D AR=2 -o src/f32-dwconv/up4x25-sse.c


################################## Unit tests #################################
tools/generate-dwconv-test.py --spec test/f32-dwconv.yaml --output test/f32-dwconv.cc
