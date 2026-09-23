#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
for MR in 1 4 6; do
  tools/xngen src/bf16-f32-gemm/neonfma-lane.c.in -D MR=${MR} -D NR=8 -o src/bf16-f32-gemm/gen/bf16-f32-gemm-${MR}x8-minmax-neonfma-lane-ld64.c &
done
for MR in 1 4; do
  tools/xngen src/bf16-f32-gemm/neonfma-lane.c.in -D MR=${MR} -D NR=16 -o src/bf16-f32-gemm/gen/bf16-f32-gemm-${MR}x16-minmax-neonfma-lane-ld64.c &
done

################################# ARM NEON BF16 ###############################
for MR in 1 4 6; do
  tools/xngen src/bf16-f32-gemm/c2-neonbf16-bfdot-lane-ld128.c.in -D MR=${MR} -D NR=8 -o src/bf16-f32-gemm/gen/bf16-f32-gemm-${MR}x8c2-minmax-neonbf16-bfdot-lane-ld128.c &
done
for MR in 1 4; do
  tools/xngen src/bf16-f32-gemm/c2-neonbf16-bfdot-lane-ld128.c.in -D MR=${MR} -D NR=16 -o src/bf16-f32-gemm/gen/bf16-f32-gemm-${MR}x16c2-minmax-neonbf16-bfdot-lane-ld128.c &
done

wait
