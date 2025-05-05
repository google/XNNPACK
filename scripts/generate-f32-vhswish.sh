#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

##################################### SIMD #####################################
tools/xngen src/f32-vhswish/simd.c.in -D ARCH=scalar          -D BATCH_TILES=1,2,4     -o src/f32-vhswish/gen/f32-vhswish-scalar.c &
tools/xngen src/f32-vhswish/simd.c.in -D ARCH=wasmsimd        -D BATCH_TILES=4,8,16    -o src/f32-vhswish/gen/f32-vhswish-wasmsimd.c &
tools/xngen src/f32-vhswish/simd.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=4,8,16    -o src/f32-vhswish/gen/f32-vhswish-wasmrelaxedsimd.c &
tools/xngen src/f32-vhswish/simd.c.in -D ARCH=neon            -D BATCH_TILES=4,8,16    -o src/f32-vhswish/gen/f32-vhswish-neon.c &
tools/xngen src/f32-vhswish/simd.c.in -D ARCH=sse2            -D BATCH_TILES=4,8,16    -o src/f32-vhswish/gen/f32-vhswish-sse2.c &
tools/xngen src/f32-vhswish/simd.c.in -D ARCH=avx             -D BATCH_TILES=8,16,32   -o src/f32-vhswish/gen/f32-vhswish-avx.c &
tools/xngen src/f32-vhswish/simd.c.in -D ARCH=fma3            -D BATCH_TILES=8,16,32   -o src/f32-vhswish/gen/f32-vhswish-fma3.c &
tools/xngen src/f32-vhswish/simd.c.in -D ARCH=avx512f         -D BATCH_TILES=16,32,64  -o src/f32-vhswish/gen/f32-vhswish-avx512f.c &
tools/xngen src/f32-vhswish/simd.c.in -D ARCH=hvx             -D BATCH_TILES=32,64,128 -o src/f32-vhswish/gen/f32-vhswish-hvx.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vhswish/rvv.c.in -D LMUL=1 -o src/f32-vhswish/gen/f32-vhswish-rvv-u1v.c &
tools/xngen src/f32-vhswish/rvv.c.in -D LMUL=2 -o src/f32-vhswish/gen/f32-vhswish-rvv-u2v.c &
tools/xngen src/f32-vhswish/rvv.c.in -D LMUL=4 -o src/f32-vhswish/gen/f32-vhswish-rvv-u4v.c &
tools/xngen src/f32-vhswish/rvv.c.in -D LMUL=8 -o src/f32-vhswish/gen/f32-vhswish-rvv-u8v.c &

wait
