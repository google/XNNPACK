#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/bf16-f32-igemm/scalar.c.in -D MR=1 -D NR=4 -o src/bf16-f32-igemm/gen/bf16-f32-igemm-1x4c2-minmax-scalar.c &
tools/xngen src/bf16-f32-igemm/scalar.c.in -D MR=4 -D NR=4 -o src/bf16-f32-igemm/gen/bf16-f32-igemm-4x4c2-minmax-scalar.c &

wait
