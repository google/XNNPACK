#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/bf16-vunary/neon.c.in -D OP=ABS -D BATCH_TILE=8  -o src/bf16-vunary/gen/bf16-vabs-neonbf16-u8.c &
tools/xngen src/bf16-vunary/neon.c.in -D OP=ABS -D BATCH_TILE=16 -o src/bf16-vunary/gen/bf16-vabs-neonbf16-u16.c &
tools/xngen src/bf16-vunary/neon.c.in -D OP=ABS -D BATCH_TILE=24 -o src/bf16-vunary/gen/bf16-vabs-neonbf16-u24.c &
