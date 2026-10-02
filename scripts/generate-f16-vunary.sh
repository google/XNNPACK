#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vunary/neon.c.in -D OP=ABS -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vabs-neon-u8.c &
tools/xngen src/f16-vunary/neon.c.in -D OP=ABS -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vabs-neon-u16.c &
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=ABS -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vabs-neonfp16arith-u8.c &
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=ABS -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vabs-neonfp16arith-u16.c &
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=NEG -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vneg-neonfp16arith-u8.c &
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=NEG -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vneg-neonfp16arith-u16.c &
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=SQR -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vsqr-neonfp16arith-u8.c &
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=SQR -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vsqr-neonfp16arith-u16.c &

################################# x86 128-bit #################################
tools/xngen src/f16-vunary/sse2.c.in -D OP=ABS -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vabs-sse2-u8.c &
tools/xngen src/f16-vunary/sse2.c.in -D OP=ABS -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vabs-sse2-u16.c &
tools/xngen src/f16-vunary/sse2.c.in -D OP=NEG -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vneg-sse2-u8.c &
tools/xngen src/f16-vunary/sse2.c.in -D OP=NEG -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vneg-sse2-u16.c &
tools/xngen src/f16-vunary/f16c.c.in -D OP=SQR -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vsqr-f16c-u8.c &
tools/xngen src/f16-vunary/f16c.c.in -D OP=SQR -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vsqr-f16c-u16.c &

################################# x86 256-bit #################################
tools/xngen src/f16-vunary/avx2.c.in -D OP=ABS -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vabs-avx2-u16.c &
tools/xngen src/f16-vunary/avx2.c.in -D OP=ABS -D BATCH_TILE=32 -o src/f16-vunary/gen/f16-vabs-avx2-u32.c &

################################# x86 512-bit #################################
tools/xngen src/f16-vunary/avx512skx.c.in -D OP=ABS -D BATCH_TILE=32 -o src/f16-vunary/gen/f16-vabs-avx512skx-u32.c &
tools/xngen src/f16-vunary/avx512skx.c.in -D OP=ABS -D BATCH_TILE=64 -o src/f16-vunary/gen/f16-vabs-avx512skx-u64.c &

################################# RISC-V Vector ###############################
tools/xngen src/f16-vunary/rvv.c.in -D OP=ABS -D LMUL=1 -o src/f16-vunary/gen/f16-vabs-rvv-u1v.c &
tools/xngen src/f16-vunary/rvv.c.in -D OP=ABS -D LMUL=2 -o src/f16-vunary/gen/f16-vabs-rvv-u2v.c &
tools/xngen src/f16-vunary/rvv.c.in -D OP=ABS -D LMUL=4 -o src/f16-vunary/gen/f16-vabs-rvv-u4v.c &
tools/xngen src/f16-vunary/rvv.c.in -D OP=ABS -D LMUL=8 -o src/f16-vunary/gen/f16-vabs-rvv-u8v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=ABS -D LMUL=1 -o src/f16-vunary/gen/f16-vabs-rvvfp16arith-u1v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=ABS -D LMUL=2 -o src/f16-vunary/gen/f16-vabs-rvvfp16arith-u2v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=ABS -D LMUL=4 -o src/f16-vunary/gen/f16-vabs-rvvfp16arith-u4v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=ABS -D LMUL=8 -o src/f16-vunary/gen/f16-vabs-rvvfp16arith-u8v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=NEG -D LMUL=1 -o src/f16-vunary/gen/f16-vneg-rvvfp16arith-u1v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=NEG -D LMUL=2 -o src/f16-vunary/gen/f16-vneg-rvvfp16arith-u2v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=NEG -D LMUL=4 -o src/f16-vunary/gen/f16-vneg-rvvfp16arith-u4v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=NEG -D LMUL=8 -o src/f16-vunary/gen/f16-vneg-rvvfp16arith-u8v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=SQR -D LMUL=1 -o src/f16-vunary/gen/f16-vsqr-rvvfp16arith-u1v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=SQR -D LMUL=2 -o src/f16-vunary/gen/f16-vsqr-rvvfp16arith-u2v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=SQR -D LMUL=4 -o src/f16-vunary/gen/f16-vsqr-rvvfp16arith-u4v.c &
tools/xngen src/f16-vunary/rvvfp16arith.c.in -D OP=SQR -D LMUL=8 -o src/f16-vunary/gen/f16-vsqr-rvvfp16arith-u8v.c &

################################# Hexagon HVX #################################
tools/xngen src/f16-vunary/hvx.c.in -D OP=ABS -D BATCH_TILE=64  -o src/f16-vunary/gen/f16-vabs-hvx-u64.c &
tools/xngen src/f16-vunary/hvx.c.in -D OP=ABS -D BATCH_TILE=128 -o src/f16-vunary/gen/f16-vabs-hvx-u128.c &

################################## Wasm SIMD ##################################
tools/xngen src/f16-vunary/wasmsimd.c.in -D OP=ABS -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vabs-wasmsimd-u8.c &
tools/xngen src/f16-vunary/wasmsimd.c.in -D OP=ABS -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vabs-wasmsimd-u16.c &

#################################### Scalar ###################################
tools/xngen src/f16-vunary/scalar.c.in -D OP=ABS -D BATCH_TILE=1 -o src/f16-vunary/gen/f16-vabs-scalar-u1.c &
tools/xngen src/f16-vunary/scalar.c.in -D OP=ABS -D BATCH_TILE=2 -o src/f16-vunary/gen/f16-vabs-scalar-u2.c &
tools/xngen src/f16-vunary/scalar.c.in -D OP=ABS -D BATCH_TILE=4 -o src/f16-vunary/gen/f16-vabs-scalar-u4.c &

wait
