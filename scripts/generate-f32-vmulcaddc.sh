#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-vmulcaddc/scalar.c.in -D CR=1 -D MR=2 -o src/f32-vmulcaddc/c1-scalar-x2.c

################################### ARM NEON ##################################
tools/xngen src/f32-vmulcaddc/neon.c.in -D CR=4 -D MR=2 -D FMA=1 -o src/f32-vmulcaddc/c4-neonfma-x2.c
tools/xngen src/f32-vmulcaddc/neon.c.in -D CR=4 -D MR=2 -D FMA=0 -o src/f32-vmulcaddc/c4-neon-x2.c

#################################### PSIMD ####################################
tools/xngen src/f32-vmulcaddc/psimd.c.in -D CR=4 -D MR=2 -o src/f32-vmulcaddc/c4-psimd-x2.c

################################### x86 SSE ###################################
tools/xngen src/f32-vmulcaddc/sse.c.in -D CR=4 -D MR=2 -o src/f32-vmulcaddc/c4-sse-x2.c


################################## Unit tests #################################
tools/generate-vmulcaddc-test.py --spec test/f32-vmulcaddc.yaml --output test/f32-vmulcaddc.cc
