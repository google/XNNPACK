#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
### LD128 micro-kernels
tools/xngen src/bf16-gemm/c2-neonbf16-bfdot-lane-ld128.c.in -D MR=1 -D NR=8 -o src/bf16-gemm/gen/1x8c2-minmax-neonbf16-bfdot-lane-ld128.c &
tools/xngen src/bf16-gemm/c2-neonbf16-bfdot-lane-ld128.c.in -D MR=4 -D NR=8 -o src/bf16-gemm/gen/4x8c2-minmax-neonbf16-bfdot-lane-ld128.c &
tools/xngen src/bf16-gemm/c2-neonbf16-bfdot-lane-ld128.c.in -D MR=5 -D NR=8 -o src/bf16-gemm/gen/5x8c2-minmax-neonbf16-bfdot-lane-ld128.c &
tools/xngen src/bf16-gemm/c2-neonbf16-bfdot-lane-ld128.c.in -D MR=6 -D NR=8 -o src/bf16-gemm/gen/6x8c2-minmax-neonbf16-bfdot-lane-ld128.c &

tools/xngen src/bf16-gemm/c8-neon.c.in -D MR=1 -D NR=4 -D EXTOPT=SHLAND -o src/bf16-gemm/gen/1x4c8-minmax-neonfma-shland.c &
tools/xngen src/bf16-gemm/c8-neon.c.in -D MR=2 -D NR=4 -D EXTOPT=SHLAND -o src/bf16-gemm/gen/2x4c8-minmax-neonfma-shland.c &
tools/xngen src/bf16-gemm/c8-neon.c.in -D MR=3 -D NR=4 -D EXTOPT=SHLAND -o src/bf16-gemm/gen/3x4c8-minmax-neonfma-shland.c &
tools/xngen src/bf16-gemm/c8-neon.c.in -D MR=4 -D NR=4 -D EXTOPT=SHLAND -o src/bf16-gemm/gen/4x4c8-minmax-neonfma-shland.c &
tools/xngen src/bf16-gemm/c8-neon.c.in -D MR=5 -D NR=4 -D EXTOPT=SHLAND -o src/bf16-gemm/gen/5x4c8-minmax-neonfma-shland.c &

tools/xngen src/bf16-gemm/c8-neon.c.in -D MR=1 -D NR=4 -D EXTOPT=ZIP -o src/bf16-gemm/gen/1x4c8-minmax-neonfma-zip.c &
tools/xngen src/bf16-gemm/c8-neon.c.in -D MR=2 -D NR=4 -D EXTOPT=ZIP -o src/bf16-gemm/gen/2x4c8-minmax-neonfma-zip.c &
tools/xngen src/bf16-gemm/c8-neon.c.in -D MR=3 -D NR=4 -D EXTOPT=ZIP -o src/bf16-gemm/gen/3x4c8-minmax-neonfma-zip.c &
tools/xngen src/bf16-gemm/c8-neon.c.in -D MR=4 -D NR=4 -D EXTOPT=ZIP -o src/bf16-gemm/gen/4x4c8-minmax-neonfma-zip.c &
tools/xngen src/bf16-gemm/c8-neon.c.in -D MR=5 -D NR=4 -D EXTOPT=ZIP -o src/bf16-gemm/gen/5x4c8-minmax-neonfma-zip.c &

tools/xngen src/bf16-gemm/c8-neonbf16.c.in -D MR=1 -D NR=4 -D BFOPT=BFDOT -o src/bf16-gemm/gen/1x4c8-minmax-neonbf16-bfdot.c &
tools/xngen src/bf16-gemm/c8-neonbf16.c.in -D MR=2 -D NR=4 -D BFOPT=BFDOT -o src/bf16-gemm/gen/2x4c8-minmax-neonbf16-bfdot.c &
tools/xngen src/bf16-gemm/c8-neonbf16.c.in -D MR=4 -D NR=4 -D BFOPT=BFDOT -o src/bf16-gemm/gen/4x4c8-minmax-neonbf16-bfdot.c &
tools/xngen src/bf16-gemm/c8-neonbf16.c.in -D MR=5 -D NR=4 -D BFOPT=BFDOT -o src/bf16-gemm/gen/5x4c8-minmax-neonbf16-bfdot.c &

tools/xngen src/bf16-gemm/c8-neonbf16.c.in -D MR=1 -D NR=4 -D BFOPT=BFMLAL -o src/bf16-gemm/gen/1x4c8-minmax-neonbf16-bfmlal.c &
tools/xngen src/bf16-gemm/c8-neonbf16.c.in -D MR=2 -D NR=4 -D BFOPT=BFMLAL -o src/bf16-gemm/gen/2x4c8-minmax-neonbf16-bfmlal.c &
tools/xngen src/bf16-gemm/c8-neonbf16.c.in -D MR=3 -D NR=4 -D BFOPT=BFMLAL -o src/bf16-gemm/gen/3x4c8-minmax-neonbf16-bfmlal.c &
tools/xngen src/bf16-gemm/c8-neonbf16.c.in -D MR=4 -D NR=4 -D BFOPT=BFMLAL -o src/bf16-gemm/gen/4x4c8-minmax-neonbf16-bfmlal.c &
tools/xngen src/bf16-gemm/c8-neonbf16.c.in -D MR=5 -D NR=4 -D BFOPT=BFMLAL -o src/bf16-gemm/gen/5x4c8-minmax-neonbf16-bfmlal.c &

################################## Unit tests #################################
tools/generate-gemm-test.py --spec test/bf16-gemm-minmax.yaml --output test/bf16-gemm-minmax.cc &

wait
