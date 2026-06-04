#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### SCALAR ###################################
tools/xngen src/f16-vlog/rational-1-3.c.in -D ARCH=scalar   -D BATCH_TILES=1,2,4,8      -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-scalar-rational-1-3-div.c &

##################################### SIMD #####################################
tools/xngen src/f16-vlog/rational-1-3.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=8,16,32 -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-wasmrelaxedsimd-rational-1-3-div.c &
tools/xngen src/f16-vlog/rational-1-3.c.in -D ARCH=neonfp16arith   -D BATCH_TILES=8,16,32 -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-neonfp16arith-rational-1-3-div.c &
tools/xngen src/f16-vlog/rational-1-3.c.in -D ARCH=avx512fp16      -D BATCH_TILES=32,64   -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-avx512fp16-rational-1-3-div.c &

tools/xngen src/f16-vlog/rational-1-3.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=8,16,32 -D DIV=NR  -o src/f16-vlog/gen/f16-vlog-wasmrelaxedsimd-rational-1-3-nr.c &
tools/xngen src/f16-vlog/rational-1-3.c.in -D ARCH=neonfp16arith   -D BATCH_TILES=8,16,32 -D DIV=NR  -o src/f16-vlog/gen/f16-vlog-neonfp16arith-rational-1-3-nr.c &
tools/xngen src/f16-vlog/rational-1-3.c.in -D ARCH=avx512fp16      -D BATCH_TILES=32,64   -D DIV=NR  -o src/f16-vlog/gen/f16-vlog-avx512fp16-rational-1-3-nr.c &

#################################### F32ACC ####################################
tools/xngen src/f16-vlog/f16-f32acc.c.in -D ARCH=scalar          -D BATCH_TILES=1,2,4        -D DIV=DIV -o src/f16-vlog/gen/f16-f32acc-vlog-scalar-rational-1-3-div.c &
tools/xngen src/f16-vlog/f16-f32acc.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=4,8,16       -D DIV=DIV -o src/f16-vlog/gen/f16-f32acc-vlog-wasmrelaxedsimd-rational-1-3-div.c &
tools/xngen src/f16-vlog/f16-f32acc.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=4,8,16       -D DIV=NR  -o src/f16-vlog/gen/f16-f32acc-vlog-wasmrelaxedsimd-rational-1-3-nr.c &
tools/xngen src/f16-vlog/f16-f32acc.c.in -D ARCH=neonfp16        -D BATCH_TILES=4,8,16       -D DIV=DIV -o src/f16-vlog/gen/f16-f32acc-vlog-neonfp16-rational-1-3-div.c &
tools/xngen src/f16-vlog/f16-f32acc.c.in -D ARCH=neonfp16        -D BATCH_TILES=4,8,16       -D DIV=NR  -o src/f16-vlog/gen/f16-f32acc-vlog-neonfp16-rational-1-3-nr.c &
tools/xngen src/f16-vlog/f16-f32acc.c.in -D ARCH=f16c            -D BATCH_TILES=8,16,32      -D DIV=DIV -o src/f16-vlog/gen/f16-f32acc-vlog-f16c-rational-1-3-div.c &
tools/xngen src/f16-vlog/f16-f32acc.c.in -D ARCH=f16c            -D BATCH_TILES=8,16,32      -D DIV=NR  -o src/f16-vlog/gen/f16-f32acc-vlog-f16c-rational-1-3-nr.c &
tools/xngen src/f16-vlog/f16-f32acc.c.in -D ARCH=avx512f         -D BATCH_TILES=16,32,64     -D DIV=DIV -o src/f16-vlog/gen/f16-f32acc-vlog-avx512f-rational-1-3-div.c &
tools/xngen src/f16-vlog/f16-f32acc.c.in -D ARCH=avx512f         -D BATCH_TILES=16,32,64     -D DIV=NR  -o src/f16-vlog/gen/f16-f32acc-vlog-avx512f-rational-1-3-nr.c &
tools/xngen src/f16-vlog/f16-f32acc.c.in -D ARCH=hvx             -D BATCH_TILES=32,64,128    -D DIV=DIV -o src/f16-vlog/gen/f16-f32acc-vlog-hvx-rational-1-3-div.c &
tools/xngen src/f16-vlog/f16-f32acc.c.in -D ARCH=hvx             -D BATCH_TILES=32,64,128    -D DIV=NR  -o src/f16-vlog/gen/f16-f32acc-vlog-hvx-rational-1-3-nr.c &

################################### RISC-V Vector #############################
tools/xngen src/f16-vlog/rvv-rational-1-3.c.in -D LMUL=1 -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-1-3-div-u1v.c &
tools/xngen src/f16-vlog/rvv-rational-1-3.c.in -D LMUL=2 -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-1-3-div-u2v.c &
tools/xngen src/f16-vlog/rvv-rational-1-3.c.in -D LMUL=4 -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-1-3-div-u4v.c &
tools/xngen src/f16-vlog/rvv-rational-1-3.c.in -D LMUL=8 -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-1-3-div-u8v.c &

tools/xngen src/f16-vlog/rvv-rational-1-3.c.in -D LMUL=1 -D DIV=NR  -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-1-3-nr-u1v.c &
tools/xngen src/f16-vlog/rvv-rational-1-3.c.in -D LMUL=2 -D DIV=NR  -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-1-3-nr-u2v.c &
tools/xngen src/f16-vlog/rvv-rational-1-3.c.in -D LMUL=4 -D DIV=NR  -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-1-3-nr-u4v.c &
tools/xngen src/f16-vlog/rvv-rational-1-3.c.in -D LMUL=8 -D DIV=NR  -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-1-3-nr-u8v.c &

##################################### RISC-V Vector ############################
tools/xngen src/f16-vlog/rvv-rational-3-3.c.in -D LMUL=1 -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-3-3-div-u1v.c &
tools/xngen src/f16-vlog/rvv-rational-3-3.c.in -D LMUL=2 -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-3-3-div-u2v.c &
tools/xngen src/f16-vlog/rvv-rational-3-3.c.in -D LMUL=4 -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-3-3-div-u4v.c &
tools/xngen src/f16-vlog/rvv-rational-3-3.c.in -D LMUL=8 -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-3-3-div-u8v.c &

wait
