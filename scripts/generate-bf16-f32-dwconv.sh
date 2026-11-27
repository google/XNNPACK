#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/bf16-f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-3p2c-minmax-scalar.c &
tools/xngen src/bf16-f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-4p2c-minmax-scalar.c &
tools/xngen src/bf16-f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-9p2c-minmax-scalar.c &
tools/xngen src/bf16-f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-25p2c-minmax-scalar-acc2.c &

wait
