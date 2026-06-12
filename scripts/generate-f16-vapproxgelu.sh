#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# ################################## SCALAR ###################################
tools/xngen src/f16-vapproxgelu/rational-6-4.c.in -D ARCH=scalar -D BATCH_TILES=1,2,4,8 -o src/f16-vapproxgelu/gen/f16-vapproxgelu-scalar-rational-6-4-div.c &

# ##################################### SIMD #####################################
tools/xngen src/f16-vapproxgelu/rational-6-4.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=8,16,32 -o src/f16-vapproxgelu/gen/f16-vapproxgelu-wasmrelaxedsimd-rational-6-4-div.c &
tools/xngen src/f16-vapproxgelu/rational-6-4.c.in -D ARCH=neonfp16arith   -D BATCH_TILES=8,16,32 -o src/f16-vapproxgelu/gen/f16-vapproxgelu-neonfp16arith-rational-6-4-div.c &
tools/xngen src/f16-vapproxgelu/rational-6-4.c.in -D ARCH=avx512fp16      -D BATCH_TILES=32,64   -o src/f16-vapproxgelu/gen/f16-vapproxgelu-avx512fp16-rational-6-4-div.c &

# #################################### F32ACC ####################################
tools/xngen src/f16-vapproxgelu/f16-f32acc.c.in -D ARCH=scalar          -D BATCH_TILES=1,2,4     -D DIV=DIV -o src/f16-vapproxgelu/gen/f16-f32acc-vapproxgelu-scalar-rational-6-4-div.c &

tools/xngen src/f16-vapproxgelu/f16-f32acc.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=4,8,16    -D DIV=DIV -o src/f16-vapproxgelu/gen/f16-f32acc-vapproxgelu-wasmrelaxedsimd-rational-6-4-div.c &
tools/xngen src/f16-vapproxgelu/f16-f32acc.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=4,8,16    -D DIV=NR  -o src/f16-vapproxgelu/gen/f16-f32acc-vapproxgelu-wasmrelaxedsimd-rational-6-4-nr.c &

tools/xngen src/f16-vapproxgelu/f16-f32acc.c.in -D ARCH=neonfp16        -D BATCH_TILES=4,8,16    -D DIV=DIV -o src/f16-vapproxgelu/gen/f16-f32acc-vapproxgelu-neonfp16-rational-6-4-div.c &
tools/xngen src/f16-vapproxgelu/f16-f32acc.c.in -D ARCH=neonfp16        -D BATCH_TILES=4,8,16    -D DIV=NR  -o src/f16-vapproxgelu/gen/f16-f32acc-vapproxgelu-neonfp16-rational-6-4-nr.c &

tools/xngen src/f16-vapproxgelu/f16-f32acc.c.in -D ARCH=f16c            -D BATCH_TILES=8,16,32   -D DIV=DIV -o src/f16-vapproxgelu/gen/f16-f32acc-vapproxgelu-f16c-rational-6-4-div.c &
tools/xngen src/f16-vapproxgelu/f16-f32acc.c.in -D ARCH=f16c            -D BATCH_TILES=8,16,32   -D DIV=NR  -o src/f16-vapproxgelu/gen/f16-f32acc-vapproxgelu-f16c-rational-6-4-nr.c &

tools/xngen src/f16-vapproxgelu/f16-f32acc.c.in -D ARCH=avx512f         -D BATCH_TILES=16,32,64  -D DIV=DIV -o src/f16-vapproxgelu/gen/f16-f32acc-vapproxgelu-avx512f-rational-6-4-div.c &
tools/xngen src/f16-vapproxgelu/f16-f32acc.c.in -D ARCH=avx512f         -D BATCH_TILES=16,32,64  -D DIV=NR  -o src/f16-vapproxgelu/gen/f16-f32acc-vapproxgelu-avx512f-rational-6-4-nr.c &

tools/xngen src/f16-vapproxgelu/f16-f32acc.c.in -D ARCH=hvx             -D BATCH_TILES=32,64,128 -D DIV=DIV -o src/f16-vapproxgelu/gen/f16-f32acc-vapproxgelu-hvx-rational-6-4-div.c &
tools/xngen src/f16-vapproxgelu/f16-f32acc.c.in -D ARCH=hvx             -D BATCH_TILES=32,64,128 -D DIV=NR  -o src/f16-vapproxgelu/gen/f16-f32acc-vapproxgelu-hvx-rational-6-4-nr.c &

# ################################## RISC-V Vector ############################
tools/xngen src/f16-vapproxgelu/rvvfp16arith-rational-6-4.c.in -D LMUL=1 -o src/f16-vapproxgelu/gen/f16-vapproxgelu-rvvfp16arith-rational-6-4-div-u1v.c &
tools/xngen src/f16-vapproxgelu/rvvfp16arith-rational-6-4.c.in -D LMUL=2 -o src/f16-vapproxgelu/gen/f16-vapproxgelu-rvvfp16arith-rational-6-4-div-u2v.c &
tools/xngen src/f16-vapproxgelu/rvvfp16arith-rational-6-4.c.in -D LMUL=4 -o src/f16-vapproxgelu/gen/f16-vapproxgelu-rvvfp16arith-rational-6-4-div-u4v.c &

wait
