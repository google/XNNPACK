#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-bf16-vcvt/neon.c.in -D BATCH_TILE=8  -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-neon-u8.c &
tools/xngen src/f32-bf16-vcvt/neon.c.in -D BATCH_TILE=16 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-neon-u16.c &
tools/xngen src/f32-bf16-vcvt/neon.c.in -D BATCH_TILE=24 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-neon-u24.c &
tools/xngen src/f32-bf16-vcvt/neon.c.in -D BATCH_TILE=32 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-neon-u32.c &

################################# ARM NEON BF16 ###############################
tools/xngen src/f32-bf16-vcvt/neonbf16.c.in -D BATCH_TILE=8  -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-neonbf16-u8.c &
tools/xngen src/f32-bf16-vcvt/neonbf16.c.in -D BATCH_TILE=16 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-neonbf16-u16.c &

#################################### Scalar ###################################
tools/xngen src/f32-bf16-vcvt/scalar.c.in -D BATCH_TILE=1 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-scalar-u1.c &
tools/xngen src/f32-bf16-vcvt/scalar.c.in -D BATCH_TILE=2 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-scalar-u2.c &
tools/xngen src/f32-bf16-vcvt/scalar.c.in -D BATCH_TILE=3 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-scalar-u3.c &
tools/xngen src/f32-bf16-vcvt/scalar.c.in -D BATCH_TILE=4 -o src/f32-bf16-vcvt/gen/f32-bf16-vcvt-scalar-u4.c &

wait
