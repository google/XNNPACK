#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### F16 TanH ##################################
# NEON+FP16ARITH
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=1 -D SAT=MINMAX -D DIV=DIV         -o src/math/gen/f16-tanh-aarch64-neonfp16arith-expm1minus-rr1-p3h1ts-div.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=1 -D SAT=SELECT -D DIV=RECPE       -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h1ts-recpe.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=1 -D SAT=MINMAX -D DIV=RECPEADJ    -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h1ts-recpeadj.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=1 -D SAT=MINMAX -D DIV=NR1RECPS    -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h1ts-nr1recps.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=1 -D SAT=MINMAX -D DIV=NR1RECPSADJ -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h1ts-nr1recpsadj.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=1 -D SAT=MINMAX -D DIV=NR1FMA      -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h1ts-nr1fma.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=1 -D SAT=MINMAX -D DIV=NR1FMAADJ   -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h1ts-nr1fmaadj.c &

tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D SAT=MINMAX -D DIV=DIV         -o src/math/gen/f16-tanh-aarch64-neonfp16arith-expm1minus-rr1-p3h2ts-div.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D SAT=SELECT -D DIV=RECPE       -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpe.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D SAT=MINMAX -D DIV=RECPEADJ    -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h2ts-recpeadj.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D SAT=MINMAX -D DIV=NR1RECPS    -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recps.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D SAT=MINMAX -D DIV=NR1RECPSADJ -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1recpsadj.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D SAT=MINMAX -D DIV=NR1FMA      -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fma.c &
tools/xngen src/math/f16-tanh-neonfp16arith-expm1minus.c.in -D P=3 -D H=2 -D SAT=MINMAX -D DIV=NR1FMAADJ   -o src/math/gen/f16-tanh-neonfp16arith-expm1minus-rr1-p3h2ts-nr1fmaadj.c &

# AVX
tools/xngen src/math/f16-tanh-avx-expm1minus.c.in -D P=3 -D H=2 -D DIV=DIV -D SAT=MINMAX -D FMA=0 -D AVX=1 -o src/math/gen/f16-tanh-f16c-expm1minus-rr1-p3h2ts-div.c &
tools/xngen src/math/f16-tanh-avx-expm1minus.c.in -D P=3 -D H=2 -D DIV=RCP -D SAT=SELECT -D FMA=0 -D AVX=1 -o src/math/gen/f16-tanh-f16c-expm1minus-rr1-p3h2ts-rcp.c &

tools/xngen src/math/f16-tanh-avx-expm1minus.c.in -D P=3 -D H=2 -D DIV=DIV -D SAT=MINMAX -D FMA=3 -D AVX=1 -o src/math/gen/f16-tanh-fma3-expm1minus-rr1-p3h2ts-div.c &
tools/xngen src/math/f16-tanh-avx-expm1minus.c.in -D P=3 -D H=2 -D DIV=RCP -D SAT=SELECT -D FMA=3 -D AVX=1 -o src/math/gen/f16-tanh-fma3-expm1minus-rr1-p3h2ts-rcp.c &

tools/xngen src/math/f16-tanh-avx-expm1minus.c.in -D P=3 -D H=2 -D DIV=DIV -D SAT=MINMAX -D FMA=3 -D AVX=2 -o src/math/gen/f16-tanh-avx2-expm1minus-rr1-p3h2ts-div.c &
tools/xngen src/math/f16-tanh-avx-expm1minus.c.in -D P=3 -D H=2 -D DIV=RCP -D SAT=SELECT -D FMA=3 -D AVX=2 -o src/math/gen/f16-tanh-avx2-expm1minus-rr1-p3h2ts-rcp.c &

tools/xngen src/math/f16-tanh-avx-polynomial.c.in -D P=17 -D H=8 -D FMA=0 -o src/math/gen/f16-tanh-f16c-polynomial-p17h8t2.c &
tools/xngen src/math/f16-tanh-avx-polynomial.c.in -D P=19 -D H=9 -D FMA=0 -o src/math/gen/f16-tanh-f16c-polynomial-p19h9t2.c &

tools/xngen src/math/f16-tanh-avx-polynomial.c.in -D P=17 -D H=8 -D FMA=3 -o src/math/gen/f16-tanh-fma3-polynomial-p17h8t2.c &
tools/xngen src/math/f16-tanh-avx-polynomial.c.in -D P=19 -D H=9 -D FMA=3 -o src/math/gen/f16-tanh-fma3-polynomial-p19h9t2.c &

################################### F32 TanH ##################################
# Scalar
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=4 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-p6h4ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=4 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-p6h4ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-p6h5ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=1 -D DIV=RCP -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-p6h5ps-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=RCP -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-p6h5ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-p6h5ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut4-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D DIV=RCP -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut4-p4h2ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut4-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut4-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut4-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut4-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut4-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut8-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut8-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut8-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=RCP -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut8-p4h2ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut8-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=RCP -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut8-p4h2ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut8-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=RCP -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut8-p4h3ps-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=RCP -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut8-p4h3ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut8-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=RCP -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut8-p4h3ps-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=RCP -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut8-p4h3ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut16-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut16-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut16-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D PS=0 -D DIV=RCP -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut16-p4h2ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut16-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut16-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut16-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut16-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut16-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=5 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut32-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=5 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut32-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=6 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr1-lut64-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=6 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1minus-rr2-lut64-p3h1ts-div.c &

tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=4 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-p6h4ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=4 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-p6h4ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-p6h5ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-p6h5ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut4-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut4-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut4-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut4-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut4-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut4-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut8-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut8-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut8-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut8-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut8-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut8-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut16-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut16-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut16-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut16-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut16-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut16-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut16-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut16-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=5 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut32-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=5 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut32-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=6 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr1-lut64-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=6 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=0 -o src/math/gen/f32-tanh-scalar-expm1plus-rr2-lut64-p3h1ts-div.c &

tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=4 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-p6h4ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=4 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-p6h4ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-p6h5ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=1 -D DIV=RCP -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-p6h5ps-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=RCP -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-p6h5ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-p6h5ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut4-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D DIV=RCP -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut4-p4h2ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut4-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut4-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut4-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=1 -D DIV=RCP -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut4-p4h3ps-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=RCP -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut4-p4h3ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut4-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut4-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut8-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut8-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut8-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=RCP -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut8-p4h2ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut8-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=RCP -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut8-p4h2ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut8-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=RCP -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut8-p4h3ps-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=RCP -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut8-p4h3ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut8-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut16-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut16-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut16-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D PS=0 -D DIV=RCP -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut16-p4h2ts-rcp.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut16-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut16-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut16-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut16-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut16-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=5 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut32-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=5 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut32-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=6 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr1-lut64-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=2 -D LOG2LUT=6 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1minus-rr2-lut64-p3h1ts-div.c &

tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=4 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-p6h4ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=4 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-p6h4ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-p6h5ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-p6h5ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut4-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut4-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut4-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut4-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut4-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut4-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut8-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut8-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut8-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut8-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut8-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut8-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut16-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut16-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut16-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut16-p4h2ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut16-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut16-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut16-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=4 -D P=4 -D H=3 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut16-p4h3ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=5 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut32-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=5 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut32-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=1 -D LOG2LUT=6 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr1-lut64-p3h1ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1plus.c.in -D RR=2 -D LOG2LUT=6 -D P=3 -D H=1 -D PS=0 -D DIV=DIV -D FMA=1 -D WASM=0 -o src/math/gen/f32-tanh-fma-expm1plus-rr2-lut64-p3h1ts-div.c &

# Wasm
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV -D FMA=0 -D WASM=1 -o src/math/gen/f32-tanh-wasm-expm1minus-rr1-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D FMA=0 -D WASM=1 -o src/math/gen/f32-tanh-wasm-expm1minus-rr1-lut8-p4h3ps-div.c &

# WAsm SIMD
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-min.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-pmin.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-max.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-pmax.c &

tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ps-div-abs-min.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ps-div-abs-pmin.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ps-div-nabs-max.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ps-div-nabs-pmax.c &

# NEON
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV             -D FMA=1 -o src/math/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS        -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2recps.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPSADJ     -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2recpsadj.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1RECPS1FMA    -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr1recps1fma.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1RECPS1FMAADJ -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr1recps1fmaadj.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2FMA          -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2fma.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2FMAADJ       -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2fmaadj.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS        -D FMA=0 -o src/math/gen/f32-tanh-neon-expm1minus-rr1-p6h5ts-nr2recps.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=NR2RECPS        -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h2ts-nr2recps.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=NR1RECPS1FMA    -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h2ts-nr1recps1fma.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=NR2FMA          -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h2ts-nr2fma.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=NR2RECPS        -D FMA=0 -o src/math/gen/f32-tanh-neon-expm1minus-rr2-lut8-p4h2ts-nr2recps.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV             -D FMA=1 -o src/math/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR2RECPS        -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h3ps-nr2recps.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR2RECPSADJ     -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h3ps-nr2recpsadj.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1RECPS1FMA    -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h3ps-nr1recps1fma.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1RECPS1FMAADJ -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h3ps-nr1recps1fmaadj.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR2FMA          -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h3ps-nr2fma.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR2FMAADJ       -D FMA=1 -o src/math/gen/f32-tanh-neonfma-expm1minus-rr1-lut8-p4h3ps-nr2fmaadj.c &
tools/xngen src/math/f32-tanh-neon-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR2RECPS        -D FMA=0 -o src/math/gen/f32-tanh-neon-expm1minus-rr2-lut8-p4h3ps-nr2recps.c &

# SSE
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV -D SAT=MINMAX -o src/math/gen/f32-tanh-sse2-expm1minus-rr1-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr1-p6h5ts-nr1.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr1-p6h5ts-nr2.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=NR1 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr2-lut8-p4h2ts-nr1.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=NR2 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr2-lut8-p4h2ts-nr2.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D SAT=MINMAX -o src/math/gen/f32-tanh-sse2-expm1minus-rr1-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr2-lut8-p4h3ps-nr1.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR2 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr2-lut8-p4h3ps-nr2.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR1 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr2-lut8-p4h3ts-nr1.c &
tools/xngen src/math/f32-tanh-sse-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR2 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1minus-rr2-lut8-p4h3ts-nr2.c &

# AVX
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV -D SAT=MINMAX -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-p6h5ts-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-p6h5ts-nr2.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D DIV=DIV -D SAT=MINMAX -D FMA=0 -D PERM=1 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-lut4-p4h2ts-perm-div.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=NR1 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr2-lut8-p4h2ts-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=2 -D PS=0 -D DIV=NR2 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr2-lut8-p4h2ts-nr2.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV -D SAT=MINMAX -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr1-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr2-lut8-p4h3ps-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR2 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr2-lut8-p4h3ps-nr2.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR1 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr2-lut8-p4h3ts-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR2 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1minus-rr2-lut8-p4h3ts-nr2.c &

tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV    -D SAT=MINMAX -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1    -D SAT=SELECT -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-p6h5ts-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1ADJ -D SAT=MINMAX -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-p6h5ts-nr1adj.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=DIV    -D SAT=MINMAX -D FMA=3 -D PERM=1 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-div.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=NR1ADJ -D SAT=MINMAX -D FMA=3 -D PERM=1 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV    -D SAT=MINMAX -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-lut8-p4h3ps-div.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1    -D SAT=SELECT -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-lut8-p4h3ps-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1ADJ -D SAT=MINMAX -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1minus-rr1-lut8-p4h3ps-nr1adj.c &

# AVX2
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-p6h5ts-nr1.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-p6h5ts-nr1adj.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3ps-perm-div.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1    -D SAT=SELECT -D PERM=1 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3ps-perm-nr1.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3ps-perm-nr1adj.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3ps-gather-div.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3ps-gather-nr1.c &
tools/xngen src/math/f32-tanh-avx2-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/math/gen/f32-tanh-avx2-expm1minus-rr1-lut8-p4h3ps-gather-nr1adj.c &

# AVX512
tools/xngen src/math/f32-tanh-avx512skx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV    -D PERM=0 -o src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-p6h5ts-div.c &
tools/xngen src/math/f32-tanh-avx512skx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1    -D PERM=0 -o src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-p6h5ts-nr1.c &
tools/xngen src/math/f32-tanh-avx512skx-expm1minus.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1ADJ -D PERM=0 -o src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-p6h5ts-nr1adj.c &
tools/xngen src/math/f32-tanh-avx512skx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=DIV    -D PERM=1 -o src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div.c &
tools/xngen src/math/f32-tanh-avx512skx-expm1minus.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D DIV=NR1ADJ -D PERM=1 -o src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj.c &
tools/xngen src/math/f32-tanh-avx512skx-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV    -D PERM=1 -o src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-perm-div.c &
tools/xngen src/math/f32-tanh-avx512skx-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1    -D PERM=1 -o src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-perm-nr1.c &
tools/xngen src/math/f32-tanh-avx512skx-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1ADJ -D PERM=1 -o src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-perm-nr1adj.c &
tools/xngen src/math/f32-tanh-avx512skx-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=DIV    -D PERM=0 -o src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-gather-div.c &
tools/xngen src/math/f32-tanh-avx512skx-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1    -D PERM=0 -o src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-gather-nr1.c &
tools/xngen src/math/f32-tanh-avx512skx-expm1minus.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D PS=1 -D DIV=NR1ADJ -D PERM=0 -o src/math/gen/f32-tanh-avx512skx-expm1minus-rr1-lut8-p4h3ps-gather-nr1adj.c &

############################### Evaluation tests ##############################
tools/generate-tanh-eval.py --spec eval/f16-tanh.yaml --output eval/f16-tanh.cc &
tools/generate-tanh-eval.py --spec eval/f32-tanh.yaml --output eval/f32-tanh.cc &

wait
