#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################ Direct FP16 ###################################
tools/xngen src/f16-vsin/poly-3.c.in -D FUN=SIN -D ARCH=scalar          -D BATCH_TILES=1,2,4,8    -o src/f16-vsin/gen/f16-vsin-scalar-poly-3.c &
tools/xngen src/f16-vsin/poly-3.c.in -D FUN=SIN -D ARCH=wasmrelaxedsimd -D BATCH_TILES=8,16,32    -o src/f16-vsin/gen/f16-vsin-wasmrelaxedsimd-poly-3.c &
tools/xngen src/f16-vsin/poly-3.c.in -D FUN=SIN -D ARCH=neonfp16arith   -D BATCH_TILES=8,16,32    -o src/f16-vsin/gen/f16-vsin-neonfp16arith-poly-3.c &
tools/xngen src/f16-vsin/poly-3.c.in -D FUN=SIN -D ARCH=avx512fp16      -D BATCH_TILES=32,64,128  -o src/f16-vsin/gen/f16-vsin-avx512fp16-poly-3.c &
tools/xngen src/f16-vsin/rvv-poly-3.c.in -D FUN=SIN -D LMUL=1 -o src/f16-vsin/gen/f16-vsin-rvvfp16arith-poly-3-u1v.c &
tools/xngen src/f16-vsin/rvv-poly-3.c.in -D FUN=SIN -D LMUL=2 -o src/f16-vsin/gen/f16-vsin-rvvfp16arith-poly-3-u2v.c &
tools/xngen src/f16-vsin/rvv-poly-3.c.in -D FUN=SIN -D LMUL=4 -o src/f16-vsin/gen/f16-vsin-rvvfp16arith-poly-3-u4v.c &
tools/xngen src/f16-vsin/rvv-poly-3.c.in -D FUN=SIN -D LMUL=8 -o src/f16-vsin/gen/f16-vsin-rvvfp16arith-poly-3-u8v.c &

############################### FP32 Accumulation ##############################
tools/xngen src/f16-vsin/f16-f32-poly-3.c.in -D FUN=SIN -D ARCH=scalar          -D BATCH_TILES=1,2,4     -o src/f16-vsin/gen/f16-f32acc-vsin-scalar-poly-3.c &
tools/xngen src/f16-vsin/f16-f32-poly-3.c.in -D FUN=SIN -D ARCH=wasmrelaxedsimd -D BATCH_TILES=4,8,16    -o src/f16-vsin/gen/f16-f32acc-vsin-wasmrelaxedsimd-poly-3.c &
tools/xngen src/f16-vsin/f16-f32-poly-3.c.in -D FUN=SIN -D ARCH=neonfp16        -D BATCH_TILES=4,8,16    -o src/f16-vsin/gen/f16-f32acc-vsin-neonfp16-poly-3.c &
tools/xngen src/f16-vsin/f16-f32-poly-3.c.in -D FUN=SIN -D ARCH=f16c            -D BATCH_TILES=8,16,32   -o src/f16-vsin/gen/f16-f32acc-vsin-f16c-poly-3.c &
tools/xngen src/f16-vsin/f16-f32-poly-3.c.in -D FUN=SIN -D ARCH=avx512f         -D BATCH_TILES=16,32,64  -o src/f16-vsin/gen/f16-f32acc-vsin-avx512f-poly-3.c &
tools/xngen src/f16-vsin/f16-f32-poly-3.c.in -D FUN=SIN -D ARCH=hvx             -D BATCH_TILES=32,64,128 -o src/f16-vsin/gen/f16-f32acc-vsin-hvx-poly-3.c &

##################################### SIMD #####################################
tools/xngen src/f16-vsin/rational-3-2.c.in -D FUN=SIN -D ARCH=scalar        -D BATCH_TILES=1,2,4,8   -D DIV=DIV -o src/f16-vsin/gen/f16-vsin-scalar-rational-3-2-div.c &
tools/xngen src/f16-vsin/rational-3-2.c.in -D FUN=SIN -D ARCH=wasmrelaxedsimd -D BATCH_TILES=8,16,32 -D DIV=DIV -o src/f16-vsin/gen/f16-vsin-wasmrelaxedsimd-rational-3-2-div.c &
tools/xngen src/f16-vsin/rational-3-2.c.in -D FUN=SIN -D ARCH=neonfp16arith -D BATCH_TILES=8,16,32   -D DIV=DIV -o src/f16-vsin/gen/f16-vsin-neonfp16arith-rational-3-2-div.c &
tools/xngen src/f16-vsin/rational-3-2.c.in -D FUN=SIN -D ARCH=avx512fp16    -D BATCH_TILES=32,64,96  -D DIV=DIV -o src/f16-vsin/gen/f16-vsin-avx512fp16-rational-3-2-div.c &

################################### RISC-V Vector ##############################
tools/xngen src/f16-vsin/rvv-rational-3-2.c.in -D FUN=SIN -D LMUL=1 -o src/f16-vsin/gen/f16-vsin-rvvfp16arith-rational-3-2-div-u1v.c &
tools/xngen src/f16-vsin/rvv-rational-3-2.c.in -D FUN=SIN -D LMUL=2 -o src/f16-vsin/gen/f16-vsin-rvvfp16arith-rational-3-2-div-u2v.c &
tools/xngen src/f16-vsin/rvv-rational-3-2.c.in -D FUN=SIN -D LMUL=4 -o src/f16-vsin/gen/f16-vsin-rvvfp16arith-rational-3-2-div-u4v.c &
tools/xngen src/f16-vsin/rvv-rational-3-2.c.in -D FUN=SIN -D LMUL=8 -o src/f16-vsin/gen/f16-vsin-rvvfp16arith-rational-3-2-div-u8v.c &

wait
