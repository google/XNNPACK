#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### F32 TanH ##################################
tools/xngen src/math/f32-tanh-wasmsimd-expm1-abs.c.in  -D LOG2LUT=3 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-lut8-p4h3-div-abs-min.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-abs.c.in  -D LOG2LUT=3 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-lut8-p4h3-div-abs-pmin.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-nabs.c.in -D LOG2LUT=3 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-lut8-p4h3-div-nabs-max.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-nabs.c.in -D LOG2LUT=3 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-lut8-p4h3-div-nabs-pmax.c &

tools/xngen src/math/f32-tanh-wasmsimd-expm1-abs.c.in  -D LOG2LUT=0 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-p6h5-div-abs-min.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-abs.c.in  -D LOG2LUT=0 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-p6h5-div-abs-pmin.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-nabs.c.in -D LOG2LUT=0 -D MINMAX=WASM   -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-p6h5-div-nabs-max.c &
tools/xngen src/math/f32-tanh-wasmsimd-expm1-nabs.c.in -D LOG2LUT=0 -D MINMAX=PSEUDO -o src/math/gen/f32-tanh-wasmsimd-expm1-rr1-p6h5-div-nabs-pmax.c &

wait
