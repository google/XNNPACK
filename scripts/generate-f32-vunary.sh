#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vunary/scalar.c.in -D OP=ABS -D BATCH_TILE=1 -o src/f32-vunary/gen/f32-vabs-scalar-x1.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=ABS -D BATCH_TILE=2 -o src/f32-vunary/gen/f32-vabs-scalar-x2.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=ABS -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vabs-scalar-x4.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=NEG -D BATCH_TILE=1 -o src/f32-vunary/gen/f32-vneg-scalar-x1.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=NEG -D BATCH_TILE=2 -o src/f32-vunary/gen/f32-vneg-scalar-x2.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=NEG -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vneg-scalar-x4.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=SQR -D BATCH_TILE=1 -o src/f32-vunary/gen/f32-vsqr-scalar-x1.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=SQR -D BATCH_TILE=2 -o src/f32-vunary/gen/f32-vsqr-scalar-x2.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=SQR -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vsqr-scalar-x4.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=ABS -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vabs-wasmsimd-x4.c &
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=ABS -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vabs-wasmsimd-x8.c &
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=NEG -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vneg-wasmsimd-x4.c &
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=NEG -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vneg-wasmsimd-x8.c &
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=SQR -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vsqr-wasmsimd-x4.c &
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=SQR -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vsqr-wasmsimd-x8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vunary/neon.c.in -D OP=ABS -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vabs-neon-x4.c &
tools/xngen src/f32-vunary/neon.c.in -D OP=ABS -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vabs-neon-x8.c &
tools/xngen src/f32-vunary/neon.c.in -D OP=NEG -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vneg-neon-x4.c &
tools/xngen src/f32-vunary/neon.c.in -D OP=NEG -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vneg-neon-x8.c &
tools/xngen src/f32-vunary/neon.c.in -D OP=SQR -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vsqr-neon-x4.c &
tools/xngen src/f32-vunary/neon.c.in -D OP=SQR -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vsqr-neon-x8.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vunary/rvv.c.in -D OP=ABS -D LMUL=1 -o src/f32-vunary/gen/f32-vabs-rvv-x1v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=ABS -D LMUL=2 -o src/f32-vunary/gen/f32-vabs-rvv-x2v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=ABS -D LMUL=4 -o src/f32-vunary/gen/f32-vabs-rvv-x4v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=ABS -D LMUL=8 -o src/f32-vunary/gen/f32-vabs-rvv-x8v.c &

tools/xngen src/f32-vunary/rvv.c.in -D OP=NEG -D LMUL=1 -o src/f32-vunary/gen/f32-vneg-rvv-x1v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=NEG -D LMUL=2 -o src/f32-vunary/gen/f32-vneg-rvv-x2v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=NEG -D LMUL=4 -o src/f32-vunary/gen/f32-vneg-rvv-x4v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=NEG -D LMUL=8 -o src/f32-vunary/gen/f32-vneg-rvv-x8v.c &

tools/xngen src/f32-vunary/rvv.c.in -D OP=SQR -D LMUL=1 -o src/f32-vunary/gen/f32-vsqr-rvv-x1v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=SQR -D LMUL=2 -o src/f32-vunary/gen/f32-vsqr-rvv-x2v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=SQR -D LMUL=4 -o src/f32-vunary/gen/f32-vsqr-rvv-x4v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=SQR -D LMUL=8 -o src/f32-vunary/gen/f32-vsqr-rvv-x8v.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vunary/sse.c.in -D OP=ABS -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vabs-sse-x4.c &
tools/xngen src/f32-vunary/sse.c.in -D OP=ABS -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vabs-sse-x8.c &
tools/xngen src/f32-vunary/sse.c.in -D OP=NEG -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vneg-sse-x4.c &
tools/xngen src/f32-vunary/sse.c.in -D OP=NEG -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vneg-sse-x8.c &
tools/xngen src/f32-vunary/sse.c.in -D OP=SQR -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vsqr-sse-x4.c &
tools/xngen src/f32-vunary/sse.c.in -D OP=SQR -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vsqr-sse-x8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vunary/avx.c.in -D OP=ABS -D BATCH_TILE=8  -o src/f32-vunary/gen/f32-vabs-avx-x8.c &
tools/xngen src/f32-vunary/avx.c.in -D OP=ABS -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vabs-avx-x16.c &
tools/xngen src/f32-vunary/avx.c.in -D OP=NEG -D BATCH_TILE=8  -o src/f32-vunary/gen/f32-vneg-avx-x8.c &
tools/xngen src/f32-vunary/avx.c.in -D OP=NEG -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vneg-avx-x16.c &
tools/xngen src/f32-vunary/avx.c.in -D OP=SQR -D BATCH_TILE=8  -o src/f32-vunary/gen/f32-vsqr-avx-x8.c &
tools/xngen src/f32-vunary/avx.c.in -D OP=SQR -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vsqr-avx-x16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vunary/avx512f.c.in -D OP=ABS -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vabs-avx512f-x16.c &
tools/xngen src/f32-vunary/avx512f.c.in -D OP=ABS -D BATCH_TILE=32 -o src/f32-vunary/gen/f32-vabs-avx512f-x32.c &
tools/xngen src/f32-vunary/avx512f.c.in -D OP=NEG -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vneg-avx512f-x16.c &
tools/xngen src/f32-vunary/avx512f.c.in -D OP=NEG -D BATCH_TILE=32 -o src/f32-vunary/gen/f32-vneg-avx512f-x32.c &
tools/xngen src/f32-vunary/avx512f.c.in -D OP=SQR -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vsqr-avx512f-x16.c &
tools/xngen src/f32-vunary/avx512f.c.in -D OP=SQR -D BATCH_TILE=32 -o src/f32-vunary/gen/f32-vsqr-avx512f-x32.c &

################################## Unit tests #################################
tools/generate-vunary-test.py --spec test/f32-vabs.yaml --output test/f32-vabs.cc &
tools/generate-vunary-test.py --spec test/f32-vneg.yaml --output test/f32-vneg.cc &
tools/generate-vunary-test.py --spec test/f32-vsqr.yaml --output test/f32-vsqr.cc &

wait
