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
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=8  -o src/f16-igemm/gen/f16-igemm-1x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=8  -o src/f16-igemm/gen/f16-igemm-4x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=8  -o src/f16-igemm/gen/f16-igemm-5x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=6 -D NR=8  -o src/f16-igemm/gen/f16-igemm-6x8-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=7 -D NR=8  -o src/f16-igemm/gen/f16-igemm-7x8-minmax-avx2-broadcast.c &

tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=1 -D NR=16 -o src/f16-igemm/gen/f16-igemm-1x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=3 -D NR=16 -o src/f16-igemm/gen/f16-igemm-3x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=4 -D NR=16 -o src/f16-igemm/gen/f16-igemm-4x16-minmax-avx2-broadcast.c &
tools/xngen src/f16-igemm/avx2-broadcast.c.in -D MR=5 -D NR=16 -o src/f16-igemm/gen/f16-igemm-5x16-minmax-avx2-broadcast.c &

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/f16-igemm-minmax.yaml --output test/f16-igemm-minmax.cc &

wait
