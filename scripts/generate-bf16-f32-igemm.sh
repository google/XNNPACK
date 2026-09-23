#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/bf16-f32-igemm/scalar.c.in -D MR=1 -D NR=4 -o src/bf16-f32-igemm/gen/bf16-f32-igemm-1x4c2-minmax-scalar.c &
tools/xngen src/bf16-f32-igemm/scalar.c.in -D MR=4 -D NR=4 -o src/bf16-f32-igemm/gen/bf16-f32-igemm-4x4c2-minmax-scalar.c &

################################### ARM NEON ##################################
for MR in 1 4 6; do
  tools/xngen src/bf16-f32-igemm/neonfma-lane.c.in -D MR=${MR} -D NR=8 -o src/bf16-f32-igemm/gen/bf16-f32-igemm-${MR}x8-minmax-neonfma-lane-ld64.c &
done
for MR in 1 4; do
  tools/xngen src/bf16-f32-igemm/neonfma-lane.c.in -D MR=${MR} -D NR=16 -o src/bf16-f32-igemm/gen/bf16-f32-igemm-${MR}x16-minmax-neonfma-lane-ld64.c &
done

################################## AVX512BF16 #################################
for MR in 1 4 5 6 7 8; do
  tools/xngen src/bf16-f32-igemm/avx512bf16-broadcast.c.in -D MR=${MR} -D NR=16 -o src/bf16-f32-igemm/gen/bf16-f32-igemm-${MR}x16c2-minmax-avx512bf16-broadcast.c &
  tools/xngen src/bf16-f32-igemm/avx512bf16-broadcast.c.in -D MR=${MR} -D NR=32 -o src/bf16-f32-igemm/gen/bf16-f32-igemm-${MR}x32c2-minmax-avx512bf16-broadcast.c &
done

wait
