#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-vunary/scalar.c.in -D OP=ABS -D BATCH_TILE=1 -o src/f32-vunary/gen/f32-vabs-scalar-u1.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=ABS -D BATCH_TILE=2 -o src/f32-vunary/gen/f32-vabs-scalar-u2.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=ABS -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vabs-scalar-u4.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=NEG -D BATCH_TILE=1 -o src/f32-vunary/gen/f32-vneg-scalar-u1.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=NEG -D BATCH_TILE=2 -o src/f32-vunary/gen/f32-vneg-scalar-u2.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=NEG -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vneg-scalar-u4.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=SQR -D BATCH_TILE=1 -o src/f32-vunary/gen/f32-vsqr-scalar-u1.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=SQR -D BATCH_TILE=2 -o src/f32-vunary/gen/f32-vsqr-scalar-u2.c &
tools/xngen src/f32-vunary/scalar.c.in -D OP=SQR -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vsqr-scalar-u4.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=ABS -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vabs-wasmsimd-u4.c &
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=ABS -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vabs-wasmsimd-u8.c &
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=NEG -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vneg-wasmsimd-u4.c &
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=NEG -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vneg-wasmsimd-u8.c &
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=SQR -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vsqr-wasmsimd-u4.c &
tools/xngen src/f32-vunary/wasmsimd.c.in -D OP=SQR -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vsqr-wasmsimd-u8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-vunary/neon.c.in -D OP=ABS -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vabs-neon-u4.c &
tools/xngen src/f32-vunary/neon.c.in -D OP=ABS -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vabs-neon-u8.c &
tools/xngen src/f32-vunary/neon.c.in -D OP=NEG -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vneg-neon-u4.c &
tools/xngen src/f32-vunary/neon.c.in -D OP=NEG -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vneg-neon-u8.c &
tools/xngen src/f32-vunary/neon.c.in -D OP=SQR -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vsqr-neon-u4.c &
tools/xngen src/f32-vunary/neon.c.in -D OP=SQR -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vsqr-neon-u8.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-vunary/rvv.c.in -D OP=ABS -D LMUL=1 -o src/f32-vunary/gen/f32-vabs-rvv-u1v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=ABS -D LMUL=2 -o src/f32-vunary/gen/f32-vabs-rvv-u2v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=ABS -D LMUL=4 -o src/f32-vunary/gen/f32-vabs-rvv-u4v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=ABS -D LMUL=8 -o src/f32-vunary/gen/f32-vabs-rvv-u8v.c &

tools/xngen src/f32-vunary/rvv.c.in -D OP=NEG -D LMUL=1 -o src/f32-vunary/gen/f32-vneg-rvv-u1v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=NEG -D LMUL=2 -o src/f32-vunary/gen/f32-vneg-rvv-u2v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=NEG -D LMUL=4 -o src/f32-vunary/gen/f32-vneg-rvv-u4v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=NEG -D LMUL=8 -o src/f32-vunary/gen/f32-vneg-rvv-u8v.c &

tools/xngen src/f32-vunary/rvv.c.in -D OP=SQR -D LMUL=1 -o src/f32-vunary/gen/f32-vsqr-rvv-u1v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=SQR -D LMUL=2 -o src/f32-vunary/gen/f32-vsqr-rvv-u2v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=SQR -D LMUL=4 -o src/f32-vunary/gen/f32-vsqr-rvv-u4v.c &
tools/xngen src/f32-vunary/rvv.c.in -D OP=SQR -D LMUL=8 -o src/f32-vunary/gen/f32-vsqr-rvv-u8v.c &

################################# x86 128-bit #################################
tools/xngen src/f32-vunary/sse.c.in -D OP=ABS -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vabs-sse-u4.c &
tools/xngen src/f32-vunary/sse.c.in -D OP=ABS -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vabs-sse-u8.c &
tools/xngen src/f32-vunary/sse.c.in -D OP=NEG -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vneg-sse-u4.c &
tools/xngen src/f32-vunary/sse.c.in -D OP=NEG -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vneg-sse-u8.c &
tools/xngen src/f32-vunary/sse.c.in -D OP=SQR -D BATCH_TILE=4 -o src/f32-vunary/gen/f32-vsqr-sse-u4.c &
tools/xngen src/f32-vunary/sse.c.in -D OP=SQR -D BATCH_TILE=8 -o src/f32-vunary/gen/f32-vsqr-sse-u8.c &

################################# x86 256-bit #################################
tools/xngen src/f32-vunary/avx.c.in -D OP=ABS -D BATCH_TILE=8  -o src/f32-vunary/gen/f32-vabs-avx-u8.c &
tools/xngen src/f32-vunary/avx.c.in -D OP=ABS -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vabs-avx-u16.c &
tools/xngen src/f32-vunary/avx.c.in -D OP=NEG -D BATCH_TILE=8  -o src/f32-vunary/gen/f32-vneg-avx-u8.c &
tools/xngen src/f32-vunary/avx.c.in -D OP=NEG -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vneg-avx-u16.c &
tools/xngen src/f32-vunary/avx.c.in -D OP=SQR -D BATCH_TILE=8  -o src/f32-vunary/gen/f32-vsqr-avx-u8.c &
tools/xngen src/f32-vunary/avx.c.in -D OP=SQR -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vsqr-avx-u16.c &

################################# x86 512-bit #################################
tools/xngen src/f32-vunary/avx512f.c.in -D OP=ABS -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vabs-avx512f-u16.c &
tools/xngen src/f32-vunary/avx512f.c.in -D OP=ABS -D BATCH_TILE=32 -o src/f32-vunary/gen/f32-vabs-avx512f-u32.c &
tools/xngen src/f32-vunary/avx512f.c.in -D OP=NEG -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vneg-avx512f-u16.c &
tools/xngen src/f32-vunary/avx512f.c.in -D OP=NEG -D BATCH_TILE=32 -o src/f32-vunary/gen/f32-vneg-avx512f-u32.c &
tools/xngen src/f32-vunary/avx512f.c.in -D OP=SQR -D BATCH_TILE=16 -o src/f32-vunary/gen/f32-vsqr-avx512f-u16.c &
tools/xngen src/f32-vunary/avx512f.c.in -D OP=SQR -D BATCH_TILE=32 -o src/f32-vunary/gen/f32-vsqr-avx512f-u32.c &

wait
