#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/u32-f32-vcvt/simd.c.in -D BATCH_TILES=4,8,12,16 -D ARCH=neon -o src/u32-f32-vcvt/gen/u32-f32-vcvt-neon.c

################################# x86 AVX2 #################################
tools/xngen src/u32-f32-vcvt/simd.c.in -D BATCH_TILES=8,16,24,32 -D ARCH=avx2 -o src/u32-f32-vcvt/gen/u32-f32-vcvt-avx2.c

################################# x86 AVX512 #################################
tools/xngen src/u32-f32-vcvt/simd.c.in -D BATCH_TILES=16,32,48,64 -D ARCH=avx512f  -o src/u32-f32-vcvt/gen/u32-f32-vcvt-avx512f.c

################################## WAsm SIMD ##################################
tools/xngen src/u32-f32-vcvt/simd.c.in -D BATCH_TILES=4,8,12,16 -D ARCH=wasmsimd -o src/u32-f32-vcvt/gen/u32-f32-vcvt-wasmsimd.c

#################################### Scalar ###################################
tools/xngen src/u32-f32-vcvt/simd.c.in -D BATCH_TILES=1,2,3,4 -D ARCH=scalar -o src/u32-f32-vcvt/gen/u32-f32-vcvt-scalar.c

wait
