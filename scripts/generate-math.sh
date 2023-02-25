#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### F32 TanH ##################################
# Scalar
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=4 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-p6h4-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=4 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-p6h4-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-p6h5-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut4-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut4-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut4-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut4-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut8-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut8-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut8-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut8-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut16-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut16-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut16-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut16-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut16-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut16-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=5 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut32-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=5 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut32-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=6 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut64-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=6 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut64-p3h1-div.c &

tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=4 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-p6h4-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=4 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-p6h4-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-p6h5-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut4-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut4-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut4-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut4-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut8-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut8-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut8-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut8-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut16-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut16-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut16-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut16-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut16-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut16-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=5 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut32-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=5 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut32-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=6 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut64-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=6 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut64-p3h1-div.c &

tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=4 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-p6h4-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=4 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-p6h4-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-p6h5-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut4-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut4-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut4-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut4-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut8-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut8-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut8-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut8-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut16-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut16-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut16-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut16-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut16-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut16-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=5 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut32-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=5 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut32-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=6 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut64-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=6 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut64-p3h1-div.c &

tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=4 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-p6h4-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=4 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-p6h4-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-p6h5-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut4-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut4-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut4-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut4-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut8-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut8-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut8-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut8-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut16-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut16-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut16-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut16-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut16-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut16-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=5 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut32-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=5 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut32-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=6 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut64-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=6 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut64-p3h1-div.c &

# NEON
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=DIV             -D FMA=1 -o src/math/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR2RECPS        -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5-nr2recps.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1RECPS1FMA    -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5-nr1recps1fma.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1RECPS1FMAADJ -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5-nr1recps1fmaadj.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR2FMA          -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5-nr2fma.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR2FMAADJ       -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5-nr2fmaadj.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR2RECPS        -D FMA=0 -o src/math/gen/f32-tanh-neon-expm1minus-rr1-p6h5-nr2recps.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=DIV             -D FMA=1 -o src/math/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR2RECPS        -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h3-nr2recps.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR1RECPS1FMA    -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h3-nr1recps1fma.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR1RECPS1FMAADJ -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h3-nr1recps1fmaadj.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR2FMA          -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h3-nr2fma.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR2FMAADJ       -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h3-nr2fmaadj.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR2RECPS        -D FMA=0 -o src/math/gen/f32-tanh-neon-expm1minus-rr1-lut8-p4h3-nr2recps.c &

# SSE
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=DIV -D SAT=MINMAX -o src/math/gen/f32-tanh-sse2-expm1minus-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr1-p6h5-nr1.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR2 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr1-p6h5-nr2.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=DIV -D SAT=MINMAX -o src/math/gen/f32-tanh-sse2-expm1minus-rr1-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR1 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr1-lut8-p4h3-nr1.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR2 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr1-lut8-p4h3-nr2.c &

# AVX
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=DIV -D SAT=MINMAX -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-p6h5-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR2 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-p6h5-nr2.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D DIV=DIV -D SAT=MINMAX -D FMA=0 -D PERM=1 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-lut4-p4h2-perm-div.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D DIV=NR1 -D SAT=SELECT -D FMA=0 -D PERM=1 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-lut4-p4h2-perm-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D DIV=NR2 -D SAT=SELECT -D FMA=0 -D PERM=1 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-lut4-p4h2-perm-nr2.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=DIV -D SAT=MINMAX -D FMA=0 -D PERM=1 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-lut4-p4h3-perm-div.c &

tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=DIV    -D SAT=MINMAX -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1    -D SAT=SELECT -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-p6h5-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1ADJ -D SAT=MINMAX -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-p6h5-nr1adj.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=DIV    -D SAT=MINMAX -D FMA=3 -D PERM=1 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-lut4-p4h3-perm-div.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=NR1    -D SAT=SELECT -D FMA=3 -D PERM=1 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-lut4-p4h3-perm-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=NR1ADJ -D SAT=MINMAX -D FMA=3 -D PERM=1 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-lut4-p4h3-perm-nr1adj.c &

# AVX2
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-p6h5-nr1.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-p6h5-nr1adj.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut4-p4h3-perm-div.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=NR1    -D SAT=SELECT -D PERM=1 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut4-p4h3-perm-nr1.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut4-p4h3-perm-nr1adj.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3-perm-div.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR1    -D SAT=SELECT -D PERM=1 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3-perm-nr1.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3-perm-nr1adj.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3-gather-div.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3-gather-nr1.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3-gather-nr1adj.c &

# AVX512
tools/xngen src/math/f32-tanh-avx512f-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=DIV    -D PERM=0 -o src/math/gen/f32-tanh-avx512f-expm1minus-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-avx512f-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1    -D PERM=0 -o src/math/gen/f32-tanh-avx512f-expm1minus-rr1-p6h5-nr1.c &
tools/xngen src/math/f32-tanh-avx512f-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1ADJ -D PERM=0 -o src/math/gen/f32-tanh-avx512f-expm1minus-rr1-p6h5-nr1adj.c &
tools/xngen src/math/f32-tanh-avx512f-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=DIV    -D PERM=1 -o src/math/gen/f32-tanh-avx512f-expm1minus-rr1-lut4-p4h3-perm-div.c &
tools/xngen src/math/f32-tanh-avx512f-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=NR1    -D PERM=1 -o src/math/gen/f32-tanh-avx512f-expm1minus-rr1-lut4-p4h3-perm-nr1.c &
tools/xngen src/math/f32-tanh-avx512f-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=NR1ADJ -D PERM=1 -o src/math/gen/f32-tanh-avx512f-expm1minus-rr1-lut4-p4h3-perm-nr1adj.c &

# WAsm SIMD
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5-div-abs-min.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5-div-abs-pmin.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5-div-nabs-max.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5-div-nabs-pmax.c &

tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3-div-abs-min.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3-div-abs-pmin.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3-div-nabs-max.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3-div-nabs-pmax.c &


wait
