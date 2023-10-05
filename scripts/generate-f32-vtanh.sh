#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### Scalar ####################################
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=1 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-p6h5ts-div-u1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=2 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-p6h5ts-div-u2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-p6h5ts-div-u4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=1 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-lut8-p4h3ts-div-u1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=2 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-lut8-p4h3ts-div-u2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=4 -D FMA=0 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-scalar-expm1minus-rr1-lut8-p4h3ts-div-u4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=1 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-p6h5ts-div-u1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=2 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-p6h5ts-div-u2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-p6h5ts-div-u4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=1 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-lut8-p4h3ts-div-u1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=2 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-lut8-p4h3ts-div-u2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=4 -D FMA=0 -D WASM=1 -o src/f32-vtanh/gen/f32-vtanh-wasm-expm1minus-rr1-lut8-p4h3ts-div-u4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=1 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-p6h5ts-div-u1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=2 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-p6h5ts-div-u2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-p6h5ts-div-u4.c &

tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=1 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-lut8-p4h3ts-div-u1.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=2 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-lut8-p4h3ts-div-u2.c &
tools/xngen src/f32-vtanh/scalar-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=4 -D FMA=1 -D WASM=0 -o src/f32-vtanh/gen/f32-vtanh-fma-expm1minus-rr1-lut8-p4h3ts-div-u4.c &

################################### x86 SSE ###################################
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-div-u4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-div-u8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-div-u12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-div-u16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr1-u4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr1-u8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr1-u12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr1-u16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr2-u4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr2-u8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr2-u12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR2 -D SAT=SELECT -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-p6h5ts-nr2-u16.c &

tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3ts-div-u4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3ts-div-u8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3ts-div-u12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=2 -o src/f32-vtanh/gen/f32-vtanh-sse2-expm1minus-rr1-lut8-p4h3ts-div-u16.c &

tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-u4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-u8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-u12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-u16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=20 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-u20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-div-u24.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-u4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-u8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-u12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-u16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=20 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-u20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr1-u24.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=4  -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-u4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-u8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=12 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-u12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-u16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=20 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-u20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR2 -D SAT=SELECT -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-p6h5ts-nr2-u24.c &

tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=4  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-u4.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-u8.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=12 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-u12.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-u16.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=20 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-u20.c &
tools/xngen src/f32-vtanh/sse-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D SAT=MINMAX -D SSE=4 -o src/f32-vtanh/gen/f32-vtanh-sse41-expm1minus-rr1-lut8-p4h3ts-div-u24.c &

################################### x86 AVX ###################################
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-u32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-u40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-u48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-u56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-u64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-u72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-div-u80.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-u32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-u40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-u48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-u56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-u64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-u72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr1-u80.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-u32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-u40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-u48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-u56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-u64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-u72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR2 -D FMA=0 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-p6h5ts-nr2-u80.c &

tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-u32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-u40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-u48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-u56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-u64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-u72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-div-u80.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-u32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-u40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-u48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-u56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-u64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-u72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1    -D FMA=3 -D SAT=SELECT -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1-u80.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-u32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-u40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-u48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-u56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-u64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-u72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-p6h5ts-nr1adj-u80.c &

tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-perm-div-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-perm-div-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-perm-div-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-perm-div-u32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-perm-div-u40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-perm-div-u48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-perm-div-u56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-perm-div-u64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-perm-div-u72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=2 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV -D FMA=0 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut4-p4h2ts-perm-div-u80.c &

tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-div-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-div-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-div-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-div-u32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-div-u40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-div-u48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-div-u56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-div-u64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-div-u72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-div-u80.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u32.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u40.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u48.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u56.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u64.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u72.c &
tools/xngen src/f32-vtanh/avx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D FMA=3 -D SAT=MINMAX -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u80.c &

tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut8-p4h3ts-div-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut8-p4h3ts-div-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut8-p4h3ts-div-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV -D FMA=0 -o src/f32-vtanh/gen/f32-vtanh-avx-expm1minus-rr1-lut8-p4h3ts-div-u32.c &

tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-div-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-div-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-div-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-div-u32.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-nr1adj-u8.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-nr1adj-u16.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-nr1adj-u24.c &
tools/xngen src/f32-vtanh/avx-expm1minus-lut.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D FMA=3 -o src/f32-vtanh/gen/f32-vtanh-fma3-expm1minus-rr1-lut8-p4h3ts-nr1adj-u32.c &

tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-u8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-u16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-u24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-u32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-u40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-u48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-u56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-u64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-u72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-div-u80.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-u8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-u16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-u24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-u32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-u40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-u48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-u56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-u64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-u72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1    -D SAT=SELECT -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1-u80.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-u8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-u16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-u24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-u32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-u40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-u48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-u56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-u64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-u72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-p6h5ts-nr1adj-u80.c &

tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-u8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-u16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-u24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-u32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-u40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-u48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-u56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-u64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-u72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-div-u80.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u80.c &

tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-u8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-u16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-u24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-u32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-u40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-u48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-u56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-u64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-u72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-div-u80.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u80.c &

tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-u8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-u16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-u24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-u32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-u40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-u48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-u56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-u64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-u72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=DIV    -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-div-u80.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=8  -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u8.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u16.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=24 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u24.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u32.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=40 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u40.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u48.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=56 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u56.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u64.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=72 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u72.c &
tools/xngen src/f32-vtanh/avx2-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80 -D DIV=NR1ADJ -D SAT=MINMAX -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx2-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u80.c &

################################# x86 AVX-512 #################################
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-u16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-u32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-u48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-u64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-u80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=96  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-u96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=112 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-u112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=128 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-u128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=144 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-u144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=160 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-div-u160.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=16  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-u16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=32  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-u32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=48  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-u48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=64  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-u64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=80  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-u80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=96  -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-u96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=112 -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-u112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=128 -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-u128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=144 -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-u144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D BATCH_TILE=160 -D DIV=NR1    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-p6h5ts-nr1-u160.c &

tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-u16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-u32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-u48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-u64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-u80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-u96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-u112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-u128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-u144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-div-u160.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=2 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut4-p4h3ts-perm-nr1adj-u160.c &

tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-u16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-u32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-u48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-u64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-u80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-u96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-u112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-u128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-u144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=DIV    -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-div-u160.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=NR1ADJ -D PERM=1 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-perm-nr1adj-u160.c &

tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-u16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-u32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-u48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-u64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-u80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-u96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-u112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-u128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-u144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=DIV    -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-div-u160.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=16  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u16.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=32  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u32.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=48  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u48.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=64  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u64.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=80  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u80.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=96  -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u96.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=112 -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u112.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=128 -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u128.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=144 -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u144.c &
tools/xngen src/f32-vtanh/avx512skx-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D BATCH_TILE=160 -D DIV=NR1ADJ -D PERM=0 -o src/f32-vtanh/gen/f32-vtanh-avx512skx-expm1minus-rr1-lut8-p4h3ts-gather-nr1adj-u160.c &

################################### Wasm SIMD ####################################
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-min-u4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-min-u8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-min-u12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-min-u16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-pmin-u4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-pmin-u8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-pmin-u12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-abs-pmin-u16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-max-u4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-max-u8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-max-u12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-max-u16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-pmax-u4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-pmax-u8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-pmax-u12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-p6h5ts-div-nabs-pmax-u16.c &

tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-min-u4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-min-u8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-min-u12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-min-u16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-pmin-u4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-pmin-u8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-pmin-u12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-abs.c.in  -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-abs-pmin-u16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-max-u4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-max-u8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-max-u12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=WASM   -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-max-u16.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-pmax-u4.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-pmax-u8.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-pmax-u12.c &
tools/xngen src/f32-vtanh/wasmsimd-expm1minus-nabs.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D MINMAX=PSEUDO -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-expm1minus-rr1-lut8-p4h3ts-div-nabs-pmax-u16.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-aarch64-neonfma-expm1minus-rr1-lut8-p4h3ts-div-u4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-aarch64-neonfma-expm1minus-rr1-lut8-p4h3ts-div-u8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-aarch64-neonfma-expm1minus-rr1-lut8-p4h3ts-div-u12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-aarch64-neonfma-expm1minus-rr1-lut8-p4h3ts-div-u16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-aarch64-neonfma-expm1minus-rr1-p6h5ts-div-u4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-aarch64-neonfma-expm1minus-rr1-p6h5ts-div-u8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-aarch64-neonfma-expm1minus-rr1-p6h5ts-div-u12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=DIV          -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-aarch64-neonfma-expm1minus-rr1-p6h5ts-div-u16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-lut8-p4h3ts-nr1recps1fma-u4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-lut8-p4h3ts-nr1recps1fma-u8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-lut8-p4h3ts-nr1recps1fma-u12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-lut8-p4h3ts-nr1recps1fma-u16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-lut8-p4h3ts-nr2fma-u4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-lut8-p4h3ts-nr2fma-u8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-lut8-p4h3ts-nr2fma-u12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=3 -D P=4 -D H=3 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-lut8-p4h3ts-nr2fma-u16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr1recps1fma-u4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr1recps1fma-u8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr1recps1fma-u12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR1RECPS1FMA -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr1recps1fma-u16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr2fma-u4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr2fma-u8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr2fma-u12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2FMA       -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr2fma-u16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=1 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr2recps-u4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=1 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr2recps-u8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=1 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr2recps-u12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=1 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-neonfma-expm1minus-rr1-p6h5ts-nr2recps-u16.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=0 -D BATCH_TILE=4  -o src/f32-vtanh/gen/f32-vtanh-neon-expm1minus-rr1-p6h5ts-nr2recps-u4.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=0 -D BATCH_TILE=8  -o src/f32-vtanh/gen/f32-vtanh-neon-expm1minus-rr1-p6h5ts-nr2recps-u8.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=0 -D BATCH_TILE=12 -o src/f32-vtanh/gen/f32-vtanh-neon-expm1minus-rr1-p6h5ts-nr2recps-u12.c &
tools/xngen src/f32-vtanh/tanh-neon-expm1minus.c.in -D LOG2LUT=0 -D P=6 -D H=5 -D PS=0 -D DIV=NR2RECPS     -D FMA=0 -D BATCH_TILE=16 -o src/f32-vtanh/gen/f32-vtanh-neon-expm1minus-rr1-p6h5ts-nr2recps-u16.c &

wait
