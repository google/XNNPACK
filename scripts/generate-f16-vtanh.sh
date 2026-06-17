#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################ Direct FP16 ###################################
tools/xngen src/f16-vtanh/rational-5-4.c.in -D ARCH=scalar          -D BATCH_TILES=1,2,4     -D DIV=DIV -o src/f16-vtanh/gen/f16-vtanh-scalar-rational-5-4-div.c &
tools/xngen src/f16-vtanh/rational-5-4.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=8,16,32   -D DIV=DIV -o src/f16-vtanh/gen/f16-vtanh-wasmrelaxedsimd-rational-5-4-div.c &
tools/xngen src/f16-vtanh/rational-5-4.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=8,16,32   -D DIV=NR  -o src/f16-vtanh/gen/f16-vtanh-wasmrelaxedsimd-rational-5-4-nr.c &
tools/xngen src/f16-vtanh/rational-5-4.c.in -D ARCH=neonfp16arith   -D BATCH_TILES=8,16,32   -D DIV=DIV -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-rational-5-4-div.c &
tools/xngen src/f16-vtanh/rational-5-4.c.in -D ARCH=neonfp16arith   -D BATCH_TILES=8,16,32   -D DIV=NR  -o src/f16-vtanh/gen/f16-vtanh-neonfp16arith-rational-5-4-nr.c &
tools/xngen src/f16-vtanh/rational-5-4.c.in -D ARCH=avx512fp16      -D BATCH_TILES=32,64,128 -D DIV=DIV -o src/f16-vtanh/gen/f16-vtanh-avx512fp16-rational-5-4-div.c &
tools/xngen src/f16-vtanh/rational-5-4.c.in -D ARCH=avx512fp16      -D BATCH_TILES=32,64,128 -D DIV=NR  -o src/f16-vtanh/gen/f16-vtanh-avx512fp16-rational-5-4-nr.c &

############################### FP32 Accumulation ##############################
tools/xngen src/f16-vtanh/f16-f32acc-rational-5-4.c.in -D ARCH=scalar          -D BATCH_TILES=1,2,4    -D DIV=DIV -o src/f16-vtanh/gen/f16-f32acc-vtanh-scalar-rational-5-4-div.c &
tools/xngen src/f16-vtanh/f16-f32acc-rational-5-4.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=4,8,16    -D DIV=DIV -o src/f16-vtanh/gen/f16-f32acc-vtanh-wasmrelaxedsimd-rational-5-4-div.c &
tools/xngen src/f16-vtanh/f16-f32acc-rational-5-4.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=4,8,16    -D DIV=NR  -o src/f16-vtanh/gen/f16-f32acc-vtanh-wasmrelaxedsimd-rational-5-4-nr.c &
tools/xngen src/f16-vtanh/f16-f32acc-rational-5-4.c.in -D ARCH=neonfp16        -D BATCH_TILES=4,8,16    -D DIV=DIV -o src/f16-vtanh/gen/f16-f32acc-vtanh-neonfp16-rational-5-4-div.c &
tools/xngen src/f16-vtanh/f16-f32acc-rational-5-4.c.in -D ARCH=neonfp16        -D BATCH_TILES=4,8,16    -D DIV=NR  -o src/f16-vtanh/gen/f16-f32acc-vtanh-neonfp16-rational-5-4-nr.c &
tools/xngen src/f16-vtanh/f16-f32acc-rational-5-4.c.in -D ARCH=f16c            -D BATCH_TILES=8,16,32   -D DIV=DIV -o src/f16-vtanh/gen/f16-f32acc-vtanh-f16c-rational-5-4-div.c &
tools/xngen src/f16-vtanh/f16-f32acc-rational-5-4.c.in -D ARCH=f16c            -D BATCH_TILES=8,16,32   -D DIV=NR  -o src/f16-vtanh/gen/f16-f32acc-vtanh-f16c-rational-5-4-nr.c &
tools/xngen src/f16-vtanh/f16-f32acc-rational-5-4.c.in -D ARCH=avx512f         -D BATCH_TILES=16,32,64  -D DIV=DIV -o src/f16-vtanh/gen/f16-f32acc-vtanh-avx512f-rational-5-4-div.c &
tools/xngen src/f16-vtanh/f16-f32acc-rational-5-4.c.in -D ARCH=avx512f         -D BATCH_TILES=16,32,64  -D DIV=NR  -o src/f16-vtanh/gen/f16-f32acc-vtanh-avx512f-rational-5-4-nr.c &
tools/xngen src/f16-vtanh/f16-f32acc-rational-5-4.c.in -D ARCH=hvx             -D BATCH_TILES=32,64,128 -D DIV=DIV -o src/f16-vtanh/gen/f16-f32acc-vtanh-hvx-rational-5-4-div.c &
tools/xngen src/f16-vtanh/f16-f32acc-rational-5-4.c.in -D ARCH=hvx             -D BATCH_TILES=32,64,128 -D DIV=NR  -o src/f16-vtanh/gen/f16-f32acc-vtanh-hvx-rational-5-4-nr.c &


################################### RISC-V Vector ##############################
tools/xngen src/f16-vtanh/rvv-rational-5-4.c.in -D LMUL=1 -D DIV=DIV -o src/f16-vtanh/gen/f16-vtanh-rvvfp16arith-rational-5-4-div-u1v.c &
tools/xngen src/f16-vtanh/rvv-rational-5-4.c.in -D LMUL=2 -D DIV=DIV -o src/f16-vtanh/gen/f16-vtanh-rvvfp16arith-rational-5-4-div-u2v.c &
tools/xngen src/f16-vtanh/rvv-rational-5-4.c.in -D LMUL=4 -D DIV=DIV -o src/f16-vtanh/gen/f16-vtanh-rvvfp16arith-rational-5-4-div-u4v.c &
tools/xngen src/f16-vtanh/rvv-rational-5-4.c.in -D LMUL=8 -D DIV=DIV -o src/f16-vtanh/gen/f16-vtanh-rvvfp16arith-rational-5-4-div-u8v.c &

tools/xngen src/f16-vtanh/rvv-rational-5-4.c.in -D LMUL=1 -D DIV=NR -o src/f16-vtanh/gen/f16-vtanh-rvvfp16arith-rational-5-4-nr-u1v.c &
tools/xngen src/f16-vtanh/rvv-rational-5-4.c.in -D LMUL=2 -D DIV=NR -o src/f16-vtanh/gen/f16-vtanh-rvvfp16arith-rational-5-4-nr-u2v.c &
tools/xngen src/f16-vtanh/rvv-rational-5-4.c.in -D LMUL=4 -D DIV=NR -o src/f16-vtanh/gen/f16-vtanh-rvvfp16arith-rational-5-4-nr-u4v.c &
tools/xngen src/f16-vtanh/rvv-rational-5-4.c.in -D LMUL=8 -D DIV=NR -o src/f16-vtanh/gen/f16-vtanh-rvvfp16arith-rational-5-4-nr-u8v.c &

wait
