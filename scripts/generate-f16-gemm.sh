#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

############################### AArch64 assembly ##############################
tools/xngen src/f16-gemm/1x16-aarch64-neonfp16arith-ld32.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-1x16-minmax-asm-aarch64-neonfp16arith-ld32.S &
tools/xngen src/f16-gemm/1x16-aarch64-neonfp16arith-ld64.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-1x16-minmax-asm-aarch64-neonfp16arith-ld64.S &
tools/xngen src/f16-gemm/4x16-aarch64-neonfp16arith-ld32.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-4x16-minmax-asm-aarch64-neonfp16arith-ld32.S &
tools/xngen src/f16-gemm/4x16-aarch64-neonfp16arith-ld64.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-4x16-minmax-asm-aarch64-neonfp16arith-ld64.S &
tools/xngen src/f16-gemm/6x16-aarch64-neonfp16arith-ld32.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-ld32.S &
tools/xngen src/f16-gemm/6x16-aarch64-neonfp16arith-ld64.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-ld64.S &
tools/xngen src/f16-gemm/1x16-aarch64-neonfp16arith-ld32.S.in -D INC=1 -o src/f16-gemm/gen/f16-gemminc-1x16-minmax-asm-aarch64-neonfp16arith-ld32.S &
tools/xngen src/f16-gemm/4x16-aarch64-neonfp16arith-ld32.S.in -D INC=1 -o src/f16-gemm/gen/f16-gemminc-4x16-minmax-asm-aarch64-neonfp16arith-ld32.S &
tools/xngen src/f16-gemm/6x16-aarch64-neonfp16arith-ld32.S.in -D INC=1 -o src/f16-gemm/gen/f16-gemminc-6x16-minmax-asm-aarch64-neonfp16arith-ld32.S &

tools/xngen src/f16-gemm/1x8-aarch64-neonfp16arith-ld64.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-1x8-minmax-asm-aarch64-neonfp16arith-ld64.S &
tools/xngen src/f16-gemm/4x8-aarch64-neonfp16arith-ld64.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-4x8-minmax-asm-aarch64-neonfp16arith-ld64.S &
tools/xngen src/f16-gemm/6x8-aarch64-neonfp16arith-ld64.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-6x8-minmax-asm-aarch64-neonfp16arith-ld64.S &
tools/xngen src/f16-gemm/8x8-aarch64-neonfp16arith-ld64.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-8x8-minmax-asm-aarch64-neonfp16arith-ld64.S &
tools/xngen src/f16-gemm/1x8-aarch64-neonfp16arith-ld64.S.in -D INC=1 -o src/f16-gemm/gen/f16-gemminc-1x8-minmax-asm-aarch64-neonfp16arith-ld64.S &
tools/xngen src/f16-gemm/4x8-aarch64-neonfp16arith-ld64.S.in -D INC=1 -o src/f16-gemm/gen/f16-gemminc-4x8-minmax-asm-aarch64-neonfp16arith-ld64.S &
tools/xngen src/f16-gemm/6x8-aarch64-neonfp16arith-ld64.S.in -D INC=1 -o src/f16-gemm/gen/f16-gemminc-6x8-minmax-asm-aarch64-neonfp16arith-ld64.S &
tools/xngen src/f16-gemm/8x8-aarch64-neonfp16arith-ld64.S.in -D INC=1 -o src/f16-gemm/gen/f16-gemminc-8x8-minmax-asm-aarch64-neonfp16arith-ld64.S &

### Cortex A55r0 micro-kernel
tools/xngen src/f16-gemm/6x16-aarch64-neonfp16arith-cortex-a55r0.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55r0.S &

### Cortex A55 micro-kernels
tools/xngen src/f16-gemm/6x16-aarch64-neonfp16arith-cortex-a55.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55.S &
tools/xngen src/f16-gemm/6x16-aarch64-neonfp16arith-cortex-a55.S.in -D INC=1 -o src/f16-gemm/gen/f16-gemminc-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55.S &

### Cortex A75 micro-kernels
tools/xngen src/f16-gemm/6x16-aarch64-neonfp16arith-cortex-a75.S.in -D INC=0 -o src/f16-gemm/gen/f16-gemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a75.S &
tools/xngen src/f16-gemm/6x16-aarch64-neonfp16arith-cortex-a75.S.in -D INC=1 -o src/f16-gemm/gen/f16-gemminc-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a75.S &

########################## ARM NEON with FP16 compute #########################
### LD64 micro-kernels
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=1 -D NR=8  -D INC=0 -o src/f16-gemm/gen/f16-gemm-1x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=4 -D NR=8  -D INC=0 -o src/f16-gemm/gen/f16-gemm-4x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=6 -D NR=8  -D INC=0 -o src/f16-gemm/gen/f16-gemm-6x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=8 -D NR=8  -D INC=0 -o src/f16-gemm/gen/f16-gemm-8x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=1 -D NR=8  -D INC=1 -o src/f16-gemm/gen/f16-gemminc-1x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=4 -D NR=8  -D INC=1 -o src/f16-gemm/gen/f16-gemminc-4x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=6 -D NR=8  -D INC=1 -o src/f16-gemm/gen/f16-gemminc-6x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=8 -D NR=8  -D INC=1 -o src/f16-gemm/gen/f16-gemminc-8x8-minmax-neonfp16arith-ld64.c &

tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=1 -D NR=16 -D INC=0 -o src/f16-gemm/gen/f16-gemm-1x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=4 -D NR=16 -D INC=0 -o src/f16-gemm/gen/f16-gemm-4x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=6 -D NR=16 -D INC=0 -o src/f16-gemm/gen/f16-gemm-6x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=8 -D NR=16 -D INC=0 -o src/f16-gemm/gen/f16-gemm-8x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=1 -D NR=16 -D INC=1 -o src/f16-gemm/gen/f16-gemminc-1x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=4 -D NR=16 -D INC=1 -o src/f16-gemm/gen/f16-gemminc-4x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=6 -D NR=16 -D INC=1 -o src/f16-gemm/gen/f16-gemminc-6x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-gemm/neonfp16arith-ld64.c.in -D MR=8 -D NR=16 -D INC=1 -o src/f16-gemm/gen/f16-gemminc-8x16-minmax-neonfp16arith-ld64.c &

################################### x86 AVX2 ###################################
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=1 -D NR=8  -D ACCTYPE=F16 -o src/f16-gemm/gen/f16-gemm-1x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=4 -D NR=8  -D ACCTYPE=F16 -o src/f16-gemm/gen/f16-gemm-4x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=5 -D NR=8  -D ACCTYPE=F16 -o src/f16-gemm/gen/f16-gemm-5x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=6 -D NR=8  -D ACCTYPE=F16 -o src/f16-gemm/gen/f16-gemm-6x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=7 -D NR=8  -D ACCTYPE=F16 -o src/f16-gemm/gen/f16-gemm-7x8-minmax-avx2-broadcast.c &

tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=1 -D NR=16 -D ACCTYPE=F16 -o src/f16-gemm/gen/f16-gemm-1x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=3 -D NR=16 -D ACCTYPE=F16 -o src/f16-gemm/gen/f16-gemm-3x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=4 -D NR=16 -D ACCTYPE=F16 -o src/f16-gemm/gen/f16-gemm-4x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=5 -D NR=16 -D ACCTYPE=F16 -o src/f16-gemm/gen/f16-gemm-5x16-minmax-avx2-broadcast.c &

tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=1 -D NR=8  -D ACCTYPE=F32 -o src/f16-f32acc-gemm/gen/f16-f32acc-gemm-1x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=4 -D NR=8  -D ACCTYPE=F32 -o src/f16-f32acc-gemm/gen/f16-f32acc-gemm-4x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=5 -D NR=8  -D ACCTYPE=F32 -o src/f16-f32acc-gemm/gen/f16-f32acc-gemm-5x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=6 -D NR=8  -D ACCTYPE=F32 -o src/f16-f32acc-gemm/gen/f16-f32acc-gemm-6x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=7 -D NR=8  -D ACCTYPE=F32 -o src/f16-f32acc-gemm/gen/f16-f32acc-gemm-7x8-minmax-avx2-broadcast.c &

tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=1 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-gemm/gen/f16-f32acc-gemm-1x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=3 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-gemm/gen/f16-f32acc-gemm-3x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=4 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-gemm/gen/f16-f32acc-gemm-4x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-gemm/avx2-broadcast.c.in -D MR=5 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-gemm/gen/f16-f32acc-gemm-5x16-minmax-avx2-broadcast.c &

################################# x86 AVX-512 FP16 #################################
### AVX512FP16+BROADCAST micro-kernels
tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=1 -D NR=32 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-1x32-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=4 -D NR=32 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-4x32-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=5 -D NR=32 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-5x32-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=6 -D NR=32 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-6x32-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=7 -D NR=32 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-7x32-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=8 -D NR=32 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-8x32-minmax-avx512fp16-broadcast.c &

tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=1 -D NR=64 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-1x64-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=4 -D NR=64 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-4x64-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=5 -D NR=64 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-5x64-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=6 -D NR=64 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-6x64-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=7 -D NR=64 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-7x64-minmax-avx512fp16-broadcast.c &
tools/xngen src/f16-gemm/avx512fp16-broadcast.c.in -D MR=8 -D NR=64 -D DATATYPE=F16 -o src/f16-gemm/gen/f16-gemm-8x64-minmax-avx512fp16-broadcast.c &

wait
