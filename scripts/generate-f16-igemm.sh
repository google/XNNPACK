#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

########################## ARM NEON with FP16 compute #########################
### LD64 micro-kernels
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=1 -D NR=8  -o src/f16-igemm/gen/f16-igemm-1x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=4 -D NR=8  -o src/f16-igemm/gen/f16-igemm-4x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=6 -D NR=8  -o src/f16-igemm/gen/f16-igemm-6x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=8 -D NR=8  -o src/f16-igemm/gen/f16-igemm-8x8-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=1 -D NR=16 -o src/f16-igemm/gen/f16-igemm-1x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=4 -D NR=16 -o src/f16-igemm/gen/f16-igemm-4x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=6 -D NR=16 -o src/f16-igemm/gen/f16-igemm-6x16-minmax-neonfp16arith-ld64.c &
tools/xngen src/f16-igemm/neonfp16arith-ld64.c.in -D MR=8 -D NR=16 -o src/f16-igemm/gen/f16-igemm-8x16-minmax-neonfp16arith-ld64.c &

################################### x86 AVX2 ###################################
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=8  -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-1x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=8  -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-4x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=8  -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-5x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=6 -D NR=8  -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-6x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=7 -D NR=8  -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-7x8-minmax-avx2-broadcast.c &

tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=16 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-1x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=3 -D NR=16 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-3x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=16 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-4x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=16 -D ACCTYPE=F16 -o src/f16-igemm/gen/f16-igemm-5x16-minmax-avx2-broadcast.c &

tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=8  -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=8  -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=8  -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=6 -D NR=8  -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-6x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=7 -D NR=8  -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-7x8-minmax-avx2-broadcast.c &

tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-1x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=3 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-3x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-4x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=16 -D ACCTYPE=F32 -o src/f16-f32acc-igemm/gen/f16-f32acc-igemm-5x16-minmax-avx2-broadcast.c &

wait # JIT requires assembly files to be generated first.

##################################### JIT #####################################

scripts/convert-assembly-to-jit.py --no-post-op -i src/f16-igemm/f16-igemm-1x16-minmax-asm-aarch64-neonfp16arith-ld64.S -o src/f16-igemm/gen/f16-igemm-1x16-aarch64-neonfp16arith-ld64.cc &
scripts/convert-assembly-to-jit.py --no-post-op -i src/f16-igemm/f16-igemm-4x16-minmax-asm-aarch64-neonfp16arith-ld64.S -o src/f16-igemm/gen/f16-igemm-4x16-aarch64-neonfp16arith-ld64.cc &
scripts/convert-assembly-to-jit.py --no-post-op --force-prfm -i src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-ld64.S -o src/f16-igemm/gen/f16-igemm-6x16-aarch64-neonfp16arith-ld64.cc &
scripts/convert-assembly-to-jit.py --no-post-op --force-prfm -i src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55.S -o src/f16-igemm/gen/f16-igemm-6x16-aarch64-neonfp16arith-cortex-a55.cc &
scripts/convert-assembly-to-jit.py --no-post-op -i src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a55r0.S -o src/f16-igemm/gen/f16-igemm-6x16-aarch64-neonfp16arith-cortex-a55r0.cc &
scripts/convert-assembly-to-jit.py --no-post-op --force-prfm -i src/f16-igemm/f16-igemm-6x16-minmax-asm-aarch64-neonfp16arith-cortex-a75.S -o src/f16-igemm/gen/f16-igemm-6x16-aarch64-neonfp16arith-cortex-a75.cc &

wait
