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

##################################### SIMD #####################################
tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-scalar-rational-9-6-div.c &
tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-sse2-rational-9-6-div.c &
tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-wasmsimd-rational-9-6-div.c &
tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-neon-rational-9-6-div.c &
tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-avx-rational-9-6-div.c &
tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=fma3 -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-fma3-rational-9-6-div.c &
tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=DIV -o src/f32-vtanh/gen/f32-vtanh-avx512f-rational-9-6-div.c &

tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16 -D DIV=NR -o src/f32-vtanh/gen/f32-vtanh-sse2-rational-9-6-nr.c &
tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16 -D DIV=NR -o src/f32-vtanh/gen/f32-vtanh-neon-rational-9-6-nr.c &
tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32 -D DIV=NR -o src/f32-vtanh/gen/f32-vtanh-avx-rational-9-6-nr.c &
tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=fma3 -D BATCH_TILES=8,16,24,32 -D DIV=NR -o src/f32-vtanh/gen/f32-vtanh-fma3-rational-9-6-nr.c &
tools/xngen src/f32-vtanh/rational-9-6.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64 -D DIV=NR -o src/f32-vtanh/gen/f32-vtanh-avx512f-rational-9-6-nr.c &

wait
