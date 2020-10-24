#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-ibilinear-chw/scalar.c.in -D PIXEL_TILE=1 -o src/f32-ibilinear-chw/gen/scalar-p1.c
tools/xngen src/f32-ibilinear-chw/scalar.c.in -D PIXEL_TILE=2 -o src/f32-ibilinear-chw/gen/scalar-p2.c
tools/xngen src/f32-ibilinear-chw/scalar.c.in -D PIXEL_TILE=4 -o src/f32-ibilinear-chw/gen/scalar-p4.c

############################### WebAssembly SIMD ##############################
tools/xngen src/f32-ibilinear-chw/wasmsimd.c.in -D PIXEL_TILE=4 -o src/f32-ibilinear-chw/gen/wasmsimd-p4.c
tools/xngen src/f32-ibilinear-chw/wasmsimd.c.in -D PIXEL_TILE=8 -o src/f32-ibilinear-chw/gen/wasmsimd-p8.c

################################## Unit tests #################################
tools/generate-ibilinear-chw-test.py --spec test/f32-ibilinear-chw.yaml --output test/f32-ibilinear-chw.cc
