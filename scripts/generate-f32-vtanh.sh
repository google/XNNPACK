#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ####################################
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=1 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-p6h5ts-div-x1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=2 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-p6h5ts-div-x2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-p6h5ts-div-x4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=1 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-lut8-p4h3ts-div-x1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=2 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-lut8-p4h3ts-div-x2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=4 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-lut8-p4h3ts-div-x4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=1 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-p6h5ts-div-x1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=2 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-p6h5ts-div-x2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-p6h5ts-div-x4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=1 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-lut8-p4h3ts-div-x1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=2 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-lut8-p4h3ts-div-x2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=4 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-lut8-p4h3ts-div-x4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=1 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-p6h5ts-div-x1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=2 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-p6h5ts-div-x2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-p6h5ts-div-x4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=1 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-lut8-p4h3ts-div-x1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=2 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-lut8-p4h3ts-div-x2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=4 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-lut8-p4h3ts-div-x4.c &

################################### x86 SSE ###################################
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-div-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-div-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-div-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-div-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr1-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr1-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr1-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr1-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr2-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr2-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr2-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr2-x16.c &

tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3ts-div-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3ts-div-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3ts-div-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3ts-div-x16.c &

tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=20 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-x20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-x24.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=20 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-x20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-x24.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=20 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-x20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-x24.c &

tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-x4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-x8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-x12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-x16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=20 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-x20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-x24.c &

################################### x86 AVX ###################################
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-x32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-x40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-x48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-x56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-x64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-x72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-x80.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-x32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-x40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-x48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-x56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-x64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-x72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-x80.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-x32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-x40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-x48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-x56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-x64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-x72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-x80.c &

tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-x32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-x40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-x48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-x56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-x64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-x72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-x80.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-x32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-x40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-x48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-x56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-x64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-x72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-x80.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-x32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-x40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-x48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-x56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-x64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-x72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-x80.c &

tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-div-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-div-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-div-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-div-x32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-div-x40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-div-x48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-div-x56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-div-x64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-div-x72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-div-x80.c &

tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-div-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-div-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-div-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-div-x32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-div-x40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-div-x48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-div-x56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-div-x64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-div-x72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-div-x80.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-nr1adj-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-nr1adj-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-nr1adj-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-nr1adj-x32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-nr1adj-x40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-nr1adj-x48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-nr1adj-x56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-nr1adj-x64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-nr1adj-x72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-nr1adj-x80.c &

tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut8-p4h3ts-div-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut8-p4h3ts-div-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut8-p4h3ts-div-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut8-p4h3ts-div-x32.c &

tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-div-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-div-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-div-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-div-x32.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-nr1adj-x8.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-nr1adj-x16.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-nr1adj-x24.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-nr1adj-x32.c &

tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-x8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-x16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-x24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-x32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-x40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-x48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-x56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-x64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-x72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-x80.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-x8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-x16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-x24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-x32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-x40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-x48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-x56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-x64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-x72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-x80.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-x8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-x16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-x24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-x32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-x40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-x48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-x56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-x64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-x72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-x80.c &

tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-x8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-x16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-x24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-x32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-x40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-x48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-x56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-x64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-x72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-x80.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x80.c &

tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-x8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-x16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-x24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-x32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-x40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-x48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-x56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-x64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-x72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-x80.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x80.c &

tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-x8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-x16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-x24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-x32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-x40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-x48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-x56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-x64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-x72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-x80.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x80.c &

################################# x86 AVX-512 #################################
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-x16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-x32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-x48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-x64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-x80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=96  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-x96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=112 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-x112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=128 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-x128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=144 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-x144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=160 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-x160.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-x16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-x32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-x48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-x64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-x80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=96  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-x96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=112 -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-x112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=128 -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-x128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=144 -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-x144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=160 -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-x160.c &

tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-x16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-x32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-x48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-x64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-x80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-x96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-x112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-x128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-x144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-x160.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-x160.c &

tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-x16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-x32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-x48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-x64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-x80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-x96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-x112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-x128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-x144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-x160.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-x160.c &

tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-x16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-x32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-x48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-x64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-x80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-x96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-x112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-x128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-x144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-x160.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-x160.c &

################################### Wasm SIMD ####################################
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-min-x4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-min-x8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-min-x12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-min-x16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-pmin-x4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-pmin-x8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-pmin-x12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-pmin-x16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-max-x4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-max-x8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-max-x12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-max-x16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-pmax-x4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-pmax-x8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-pmax-x12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-pmax-x16.c &

tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-min-x4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-min-x8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-min-x12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-min-x16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-pmin-x4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-pmin-x8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-pmin-x12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-pmin-x16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-max-x4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-max-x8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-max-x12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-max-x16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-pmax-x4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-pmax-x8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-pmax-x12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-pmax-x16.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-lut8-p4h3ts-div-x4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-lut8-p4h3ts-div-x8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-lut8-p4h3ts-div-x12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-lut8-p4h3ts-div-x16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-p6h5ts-div-x4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-p6h5ts-div-x8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-p6h5ts-div-x12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-aarch64-neonfma-expm1minus-rr1-p6h5ts-div-x16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p4h3ts-nr1recps1fma-x4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p4h3ts-nr1recps1fma-x8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p4h3ts-nr1recps1fma-x12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p4h3ts-nr1recps1fma-x16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p4h3ts-nr2fma-x4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p4h3ts-nr2fma-x8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p4h3ts-nr2fma-x12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p4h3ts-nr2fma-x16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr1recps1fma-x4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr1recps1fma-x8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr1recps1fma-x12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr1recps1fma-x16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2fma-x4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2fma-x8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2fma-x12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2fma-x16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2recps-x4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2recps-x8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2recps-x12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-neonfma-expm1minus-rr1-p6h5ts-nr2recps-x16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=0 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-tanh-neon-expm1minus-rr1-p6h5ts-nr2recps-x4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=0 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-tanh-neon-expm1minus-rr1-p6h5ts-nr2recps-x8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=0 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-tanh-neon-expm1minus-rr1-p6h5ts-nr2recps-x12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=0 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-tanh-neon-expm1minus-rr1-p6h5ts-nr2recps-x16.c &


################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vtanh.yaml --output test/f32-vtanh.cc &

wait
