#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### F32 TanH ##################################
# Scalar
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=2 -D LOG2LUT=0 -D P=6 -D H=5 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr2-p6h5-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr1-lut4-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr1-lut4-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=3 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr1-lut8-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=2 -D LOG2LUT=3 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr2-lut8-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr1-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=2 -D LOG2LUT=3 -D P=4 -D H=3 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr2-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=4 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr1-lut16-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr1-lut16-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=5 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr1-lut32-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=6 -D P=3 -D H=1 -D FMA=0 -o src/math/gen/f32-tanh-scalar-expm1-rr1-lut64-p3h1-div.c &

tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1-rr1-lut4-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=3 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1-rr1-lut8-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1-rr1-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=4 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1-rr1-lut16-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=4 -D P=4 -D H=2 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1-rr1-lut16-p4h2-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=5 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1-rr1-lut32-p3h1-div.c &
tools/xngen src/math/f32-tanh-scalar-expm1.c.in -D RR=1 -D LOG2LUT=6 -D P=3 -D H=1 -D FMA=1 -o src/math/gen/f32-tanh-fma-expm1-rr1-lut64-p3h1-div.c &

# SSE
tools/xngen src/math/f32-tanh-sse-expm1.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=DIV -D SAT=MINMAX -o src/math/gen/f32-tanh-sse2-expm1-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-sse-expm1.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1-rr1-p6h5-nr1.c &
tools/xngen src/math/f32-tanh-sse-expm1.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR2 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1-rr1-p6h5-nr2.c &
tools/xngen src/math/f32-tanh-sse-expm1.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=DIV -D SAT=MINMAX -o src/math/gen/f32-tanh-sse2-expm1-rr1-lut8-p4h3-div.c &
tools/xngen src/math/f32-tanh-sse-expm1.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR1 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1-rr1-lut8-p4h3-nr1.c &
tools/xngen src/math/f32-tanh-sse-expm1.c.in -D RR=1 -D LOG2LUT=3 -D P=4 -D H=3 -D DIV=NR2 -D SAT=SELECT -o src/math/gen/f32-tanh-sse2-expm1-rr1-lut8-p4h3-nr2.c &

# AVX
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=DIV -D SAT=MINMAX -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1-rr1-p6h5-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR2 -D SAT=SELECT -D FMA=0 -D PERM=0 -o src/math/gen/f32-tanh-avx-expm1-rr1-p6h5-nr2.c &
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D DIV=DIV -D SAT=MINMAX -D FMA=0 -D PERM=1 -o src/math/gen/f32-tanh-avx-expm1-rr1-lut4-p4h2-perm-div.c &
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D DIV=NR1 -D SAT=SELECT -D FMA=0 -D PERM=1 -o src/math/gen/f32-tanh-avx-expm1-rr1-lut4-p4h2-perm-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=2 -D DIV=NR2 -D SAT=SELECT -D FMA=0 -D PERM=1 -o src/math/gen/f32-tanh-avx-expm1-rr1-lut4-p4h2-perm-nr2.c &
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=DIV -D SAT=MINMAX -D FMA=0 -D PERM=1 -o src/math/gen/f32-tanh-avx-expm1-rr1-lut4-p4h3-perm-div.c &

tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=DIV    -D SAT=MINMAX -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1-rr1-p6h5-div.c &
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1    -D SAT=SELECT -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1-rr1-p6h5-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=0 -D P=6 -D H=5 -D DIV=NR1ADJ -D SAT=SELECT -D FMA=3 -D PERM=0 -o src/math/gen/f32-tanh-fma3-expm1-rr1-p6h5-nr1adj.c &
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=DIV    -D SAT=MINMAX -D FMA=3 -D PERM=1 -o src/math/gen/f32-tanh-fma3-expm1-rr1-lut4-p4h3-perm-div.c &
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=NR1    -D SAT=SELECT -D FMA=3 -D PERM=1 -o src/math/gen/f32-tanh-fma3-expm1-rr1-lut4-p4h3-perm-nr1.c &
tools/xngen src/math/f32-tanh-avx-expm1.c.in -D RR=1 -D LOG2LUT=2 -D P=4 -D H=3 -D DIV=NR1ADJ -D SAT=SELECT -D FMA=3 -D PERM=1 -o src/math/gen/f32-tanh-fma3-expm1-rr1-lut4-p4h3-perm-nr1adj.c &

# WAsm SIMD
tools/xngen src/math/f32-tanh-wasmsimd-expm1-abs.c.in  -D LOG2LUT=0 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-p6h5-div-abs-min.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-abs.c.in  -D LOG2LUT=0 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-p6h5-div-abs-pmin.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-nabs.c.in -D LOG2LUT=0 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-p6h5-div-nabs-max.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-nabs.c.in -D LOG2LUT=0 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-p6h5-div-nabs-pmax.c &

tools/xngen src/math/f32-tanh-wasmsimd-expm1-abs.c.in  -D LOG2LUT=3 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-lut8-p4h3-div-abs-min.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-abs.c.in  -D LOG2LUT=3 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-lut8-p4h3-div-abs-pmin.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-nabs.c.in -D LOG2LUT=3 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-lut8-p4h3-div-nabs-max.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-nabs.c.in -D LOG2LUT=3 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-lut8-p4h3-div-nabs-pmax.c &


wait
