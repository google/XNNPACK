#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

#################################### Scalar ###################################
tools/xngen src/f32-vunary/simd.c.in -D OP=ABS -D ARCH=scalar -D BATCH_TILES="1,2,4" -o src/f32-vunary/gen/f32-vabs-scalar.c
tools/xngen src/f32-vunary/simd.c.in -D OP=NEG -D ARCH=scalar -D BATCH_TILES="1,2,4" -o src/f32-vunary/gen/f32-vneg-scalar.c
tools/xngen src/f32-vunary/simd.c.in -D OP=SQR -D ARCH=scalar -D BATCH_TILES="1,2,4" -o src/f32-vunary/gen/f32-vsqr-scalar.c

################################# x86 128-bit #################################
tools/xngen src/f32-vunary/simd.c.in -D OP=ABS -D ARCH=sse2 -D BATCH_TILES="4,8,12" -o src/f32-vunary/gen/f32-vabs-sse2.c
tools/xngen src/f32-vunary/simd.c.in -D OP=NEG -D ARCH=sse2 -D BATCH_TILES="4,8,12" -o src/f32-vunary/gen/f32-vneg-sse2.c
tools/xngen src/f32-vunary/simd.c.in -D OP=SQR -D ARCH=sse2 -D BATCH_TILES="4,8,12" -o src/f32-vunary/gen/f32-vsqr-sse2.c

################################# x86 256-bit #################################
tools/xngen src/f32-vunary/simd.c.in -D OP=ABS -D ARCH=avx -D BATCH_TILES="8,16,24" -o src/f32-vunary/gen/f32-vabs-avx.c
tools/xngen src/f32-vunary/simd.c.in -D OP=NEG -D ARCH=avx -D BATCH_TILES="8,16,24" -o src/f32-vunary/gen/f32-vneg-avx.c
tools/xngen src/f32-vunary/simd.c.in -D OP=SQR -D ARCH=avx -D BATCH_TILES="8,16,24" -o src/f32-vunary/gen/f32-vsqr-avx.c

################################# x86 512-bit #################################
tools/xngen src/f32-vunary/simd.c.in -D OP=ABS -D ARCH=avx512f -D BATCH_TILES="16,32,48" -o src/f32-vunary/gen/f32-vabs-avx512f.c
tools/xngen src/f32-vunary/simd.c.in -D OP=NEG -D ARCH=avx512f -D BATCH_TILES="16,32,48" -o src/f32-vunary/gen/f32-vneg-avx512f.c
tools/xngen src/f32-vunary/simd.c.in -D OP=SQR -D ARCH=avx512f -D BATCH_TILES="16,32,48" -o src/f32-vunary/gen/f32-vsqr-avx512f.c

################################### ARM NEON ##################################
tools/xngen src/f32-vunary/simd.c.in -D OP=ABS -D ARCH=neon -D BATCH_TILES="4,8,12" -o src/f32-vunary/gen/f32-vabs-neon.c
tools/xngen src/f32-vunary/simd.c.in -D OP=NEG -D ARCH=neon -D BATCH_TILES="4,8,12" -o src/f32-vunary/gen/f32-vneg-neon.c
tools/xngen src/f32-vunary/simd.c.in -D OP=SQR -D ARCH=neon -D BATCH_TILES="4,8,12" -o src/f32-vunary/gen/f32-vsqr-neon.c

################################## WAsm SIMD ##################################
tools/xngen src/f32-vunary/simd.c.in -D OP=ABS -D ARCH=wasmsimd -D BATCH_TILES="4,8,12" -o src/f32-vunary/gen/f32-vabs-wasmsimd.c
tools/xngen src/f32-vunary/simd.c.in -D OP=NEG -D ARCH=wasmsimd -D BATCH_TILES="4,8,12" -o src/f32-vunary/gen/f32-vneg-wasmsimd.c
tools/xngen src/f32-vunary/simd.c.in -D OP=SQR -D ARCH=wasmsimd -D BATCH_TILES="4,8,12" -o src/f32-vunary/gen/f32-vsqr-wasmsimd.c

################################## Hexagon HVX #################################
tools/xngen src/f32-vunary/simd.c.in -D OP=ABS -D ARCH=hvx -D BATCH_TILES="32,64,128" -o src/f32-vunary/gen/f32-vabs-hvx.c
tools/xngen src/f32-vunary/simd.c.in -D OP=NEG -D ARCH=hvx -D BATCH_TILES="32,64,128" -o src/f32-vunary/gen/f32-vneg-hvx.c
tools/xngen src/f32-vunary/simd.c.in -D OP=SQR -D ARCH=hvx -D BATCH_TILES="32,64,128" -o src/f32-vunary/gen/f32-vsqr-hvx.c

wait
