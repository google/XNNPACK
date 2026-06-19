#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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
