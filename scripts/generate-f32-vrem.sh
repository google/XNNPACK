#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD VREM #####################################
tools/xngen src/f32-vrem/f32-vrem.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/f32-vrem/gen/f32-vrem-scalar.c &
tools/xngen src/f32-vrem/f32-vrem.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16  -o src/f32-vrem/gen/f32-vrem-sse2.c &
tools/xngen src/f32-vrem/f32-vrem.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/f32-vrem/gen/f32-vrem-wasmsimd.c &
tools/xngen src/f32-vrem/f32-vrem.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/f32-vrem/gen/f32-vrem-neon.c &
tools/xngen src/f32-vrem/f32-vrem.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32  -o src/f32-vrem/gen/f32-vrem-avx.c &
tools/xngen src/f32-vrem/f32-vrem.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/f32-vrem/gen/f32-vrem-avx2.c &
tools/xngen src/f32-vrem/f32-vrem.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/f32-vrem/gen/f32-vrem-avx512f.c &
tools/xngen src/f32-vrem/f32-vrem.c.in -D ARCH=hvx -D BATCH_TILES=32,64  -o src/f32-vrem/gen/f32-vrem-hvx.c &

##################################### SIMD VREMC #####################################
tools/xngen src/f32-vrem/f32-vremc.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/f32-vrem/gen/f32-vremc-scalar.c &
tools/xngen src/f32-vrem/f32-vremc.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16  -o src/f32-vrem/gen/f32-vremc-sse2.c &
tools/xngen src/f32-vrem/f32-vremc.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/f32-vrem/gen/f32-vremc-wasmsimd.c &
tools/xngen src/f32-vrem/f32-vremc.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/f32-vrem/gen/f32-vremc-neon.c &
tools/xngen src/f32-vrem/f32-vremc.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32  -o src/f32-vrem/gen/f32-vremc-avx.c &
tools/xngen src/f32-vrem/f32-vremc.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/f32-vrem/gen/f32-vremc-avx2.c &
tools/xngen src/f32-vrem/f32-vremc.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/f32-vrem/gen/f32-vremc-avx512f.c &
tools/xngen src/f32-vrem/f32-vremc.c.in -D ARCH=hvx -D BATCH_TILES=32,64  -o src/f32-vrem/gen/f32-vremc-hvx.c &

##################################### SIMD VRREMC #####################################
tools/xngen src/f32-vrem/f32-vrremc.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8  -o src/f32-vrem/gen/f32-vrremc-scalar.c &
tools/xngen src/f32-vrem/f32-vrremc.c.in -D ARCH=sse2 -D BATCH_TILES=4,8,12,16  -o src/f32-vrem/gen/f32-vrremc-sse2.c &
tools/xngen src/f32-vrem/f32-vrremc.c.in -D ARCH=wasmsimd -D BATCH_TILES=4,8,12,16  -o src/f32-vrem/gen/f32-vrremc-wasmsimd.c &
tools/xngen src/f32-vrem/f32-vrremc.c.in -D ARCH=neon -D BATCH_TILES=4,8,12,16  -o src/f32-vrem/gen/f32-vrremc-neon.c &
tools/xngen src/f32-vrem/f32-vrremc.c.in -D ARCH=avx -D BATCH_TILES=8,16,24,32  -o src/f32-vrem/gen/f32-vrremc-avx.c &
tools/xngen src/f32-vrem/f32-vrremc.c.in -D ARCH=avx2 -D BATCH_TILES=8,16,24,32  -o src/f32-vrem/gen/f32-vrremc-avx2.c &
tools/xngen src/f32-vrem/f32-vrremc.c.in -D ARCH=avx512f -D BATCH_TILES=16,32,48,64  -o src/f32-vrem/gen/f32-vrremc-avx512f.c &
tools/xngen src/f32-vrem/f32-vrremc.c.in -D ARCH=hvx -D BATCH_TILES=32,64  -o src/f32-vrem/gen/f32-vrremc-hvx.c &

wait
