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

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vtanh.yaml --output test/f32-vtanh.cc &

wait
