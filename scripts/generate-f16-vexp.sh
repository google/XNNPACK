#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################ Direct FP16 ###################################
tools/xngen src/f16-vexp/poly-3.c.in -D ARCH=scalar          -D BATCH_TILES=1,2,4,8   -o src/f16-vexp/gen/f16-vexp-scalar-poly-3.c &
tools/xngen src/f16-vexp/poly-3.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=8,16,32   -o src/f16-vexp/gen/f16-vexp-wasmrelaxedsimd-poly-3.c &
tools/xngen src/f16-vexp/poly-3.c.in -D ARCH=neonfp16arith   -D BATCH_TILES=8,16,32   -o src/f16-vexp/gen/f16-vexp-neonfp16arith-poly-3.c &
tools/xngen src/f16-vexp/poly-3.c.in -D ARCH=avx512fp16      -D BATCH_TILES=32,64,128 -o src/f16-vexp/gen/f16-vexp-avx512fp16-poly-3.c &

############################### FP32 Accumulation ##############################
tools/xngen src/f16-vexp/f16-f32-poly-3.c.in -D ARCH=scalar          -D BATCH_TILES=1,2,4     -o src/f16-vexp/gen/f16-f32acc-vexp-scalar-poly-3.c &
tools/xngen src/f16-vexp/f16-f32-poly-3.c.in -D ARCH=wasmrelaxedsimd -D BATCH_TILES=4,8,16    -o src/f16-vexp/gen/f16-f32acc-vexp-wasmrelaxedsimd-poly-3.c &
tools/xngen src/f16-vexp/f16-f32-poly-3.c.in -D ARCH=neonfp16        -D BATCH_TILES=4,8,16    -o src/f16-vexp/gen/f16-f32acc-vexp-neonfp16-poly-3.c &
tools/xngen src/f16-vexp/f16-f32-poly-3.c.in -D ARCH=f16c            -D BATCH_TILES=8,16,32   -o src/f16-vexp/gen/f16-f32acc-vexp-f16c-poly-3.c &
tools/xngen src/f16-vexp/f16-f32-poly-3.c.in -D ARCH=avx512f         -D BATCH_TILES=16,32,64  -o src/f16-vexp/gen/f16-f32acc-vexp-avx512f-poly-3.c &
tools/xngen src/f16-vexp/f16-f32-poly-3.c.in -D ARCH=hvx             -D BATCH_TILES=32,64,128 -o src/f16-vexp/gen/f16-f32acc-vexp-hvx-poly-3.c &

################################### RISC-V Vector ##############################
tools/xngen src/f16-vexp/rvv-poly-3.c.in -D LMUL=1 -o src/f16-vexp/gen/f16-vexp-rvvfp16arith-poly-3-u1v.c &
tools/xngen src/f16-vexp/rvv-poly-3.c.in -D LMUL=2 -o src/f16-vexp/gen/f16-vexp-rvvfp16arith-poly-3-u2v.c &
tools/xngen src/f16-vexp/rvv-poly-3.c.in -D LMUL=4 -o src/f16-vexp/gen/f16-vexp-rvvfp16arith-poly-3-u4v.c &
tools/xngen src/f16-vexp/rvv-poly-3.c.in -D LMUL=8 -o src/f16-vexp/gen/f16-vexp-rvvfp16arith-poly-3-u8v.c &

wait
