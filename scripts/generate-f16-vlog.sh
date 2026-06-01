#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### SCALAR ###################################
tools/xngen src/f16-vlog/rational-3-3.c.in -D ARCH=scalar   -D BATCH_TILES=1,2,4,8      -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-scalar-rational-3-3-div.c &

##################################### SIMD #####################################
tools/xngen src/f16-vlog/rational-3-3.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-wasmrelaxedsimd-rational-3-3-div.c &
tools/xngen src/f16-vlog/rational-3-3.c.in -D ARCH=neonfp16arith   -D BATCH_TILES=8,16,24,32 -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-neonfp16arith-rational-3-3-div.c &
tools/xngen src/f16-vlog/rational-3-3.c.in -D ARCH=avx512fp16      -D BATCH_TILES=32,64,96   -D DIV=DIV -o src/f16-vlog/gen/f16-vlog-avx512fp16-rational-3-3-div.c &

##################################### RISC-V Vector ############################
tools/xngen src/f16-vlog/rvv-rational-3-3.c.in -D LMUL=1 -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-3-3-div-u1v.c &
tools/xngen src/f16-vlog/rvv-rational-3-3.c.in -D LMUL=2 -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-3-3-div-u2v.c &
tools/xngen src/f16-vlog/rvv-rational-3-3.c.in -D LMUL=4 -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-3-3-div-u4v.c &
tools/xngen src/f16-vlog/rvv-rational-3-3.c.in -D LMUL=8 -o src/f16-vlog/gen/f16-vlog-rvvfp16arith-rational-3-3-div-u8v.c &

wait
