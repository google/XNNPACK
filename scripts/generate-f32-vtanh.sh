#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ####################################
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=1 -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-p6h5-div-x1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=2 -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-p6h5-div-x2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=4 -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-p6h5-div-x4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=1 -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-lut8-p4h3-div-x1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=2 -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-lut8-p4h3-div-x2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=4 -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-lut8-p4h3-div-x4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=1 -D FMA=1 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-p6h5-div-x1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=2 -D FMA=1 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-p6h5-div-x2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=4 -D FMA=1 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-p6h5-div-x4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=1 -D FMA=1 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-lut8-p4h3-div-x1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=2 -D FMA=1 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-lut8-p4h3-div-x2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=4 -D FMA=1 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-lut8-p4h3-div-x4.c &

################################### x86 SSE ###################################
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-div-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-div-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-div-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-div-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=4  -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-nr1-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=8  -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-nr1-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=12 -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-nr1-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=16 -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-nr1-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=4  -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-nr2-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=8  -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-nr2-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=12 -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-nr2-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=16 -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5-nr2-x16.c &

tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3-div-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3-div-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3-div-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3-div-x16.c &

tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-div-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-div-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-div-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-div-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=20 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-div-x20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-div-x24.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=4  -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr1-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=8  -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr1-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=12 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr1-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=16 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr1-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=20 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr1-x20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=24 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr1-x24.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=4  -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr2-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=8  -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr2-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=12 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr2-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=16 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr2-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=20 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr2-x20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D BATCH_TILE=24 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5-nr2-x24.c &

tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3-div-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3-div-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3-div-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3-div-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=20 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3-div-x20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3-div-x24.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vtanh.yaml --output test/f32-vtanh.cc &

wait
