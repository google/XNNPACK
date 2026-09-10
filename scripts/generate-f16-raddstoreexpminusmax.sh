#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Scalar
tools/xngen src/f16-raddstoreexpminusmax/rr2-p2.c.in -D ARCH=scalar -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-scalar-rr2-p2-u1.c &

# Portable FP16 arithmetic
tools/xngen src/f16-raddstoreexpminusmax/rr2-p2.c.in -D ARCH=avx512fp16 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx512fp16-rr2-p2-u32.c &

# ARM NEON+FP16ARITH
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u16.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u16-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u32.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u32-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/neonfp16arith-rr2-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-neonfp16arith-rr2-p2-u32-acc4.c &

# x86 AVX2
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u16.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u16-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u32.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u32-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u32-acc4.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=40 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u40.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=40 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u40-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=40 -D ACCUMULATORS=5 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u40-acc5.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=48 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u48.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=48 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u48-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=48 -D ACCUMULATORS=3 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u48-acc3.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=64 -D ACCUMULATORS=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u64.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u64-acc2.c &
tools/xngen src/f16-raddstoreexpminusmax/avx2-rr1-p2.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-avx2-rr1-p2-u64-acc4.c &

# RISC-V Vector
tools/xngen src/f16-raddstoreexpminusmax/rvvfp16arith-rr2-p2.c.in -D LMUL=1 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-rvvfp16arith-rr2-p2-u1v.c &
tools/xngen src/f16-raddstoreexpminusmax/rvvfp16arith-rr2-p2.c.in -D LMUL=2 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-rvvfp16arith-rr2-p2-u2v.c &
tools/xngen src/f16-raddstoreexpminusmax/rvvfp16arith-rr2-p2.c.in -D LMUL=4 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-rvvfp16arith-rr2-p2-u4v.c &

# WebAssembly Relaxed SIMD FP16
tools/xngen src/f16-raddstoreexpminusmax/rr2-p2.c.in -D ARCH=wasmrelaxedsimdfp16 -o src/f16-raddstoreexpminusmax/gen/f16-raddstoreexpminusmax-wasmrelaxedsimdfp16-rr2-p2-u8.c &

# WebAssembly Relaxed SIMD with FP32 accumulation
tools/xngen src/f16-raddstoreexpminusmax/f16-f32acc-rr2-p5.c.in -D ARCH=wasmrelaxedsimd -o src/f16-raddstoreexpminusmax/gen/f16-f32acc-raddstoreexpminusmax-wasmrelaxedsimd-rr2-p5-u4.c &

# Portable FP32 accumulation
tools/xngen src/f16-raddstoreexpminusmax/f16-f32acc-rr2-p5.c.in -D ARCH=f16c -o src/f16-raddstoreexpminusmax/gen/f16-f32acc-raddstoreexpminusmax-f16c-rr2-p5-u8.c &
tools/xngen src/f16-raddstoreexpminusmax/f16-f32acc-rr2-p5.c.in -D ARCH=avx512f -o src/f16-raddstoreexpminusmax/gen/f16-f32acc-raddstoreexpminusmax-avx512f-rr2-p5-u16.c &

wait
