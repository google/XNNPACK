#!/bin/sh
# Copyright 2022 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=ABS -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vabs-neonfp16arith-u8.c &
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=ABS -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vabs-neonfp16arith-u16.c &
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=NEG -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vneg-neonfp16arith-u8.c &
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=NEG -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vneg-neonfp16arith-u16.c &
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=SQR -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vsqr-neonfp16arith-u8.c &
tools/xngen src/f16-vunary/neonfp16arith.c.in -D OP=SQR -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vsqr-neonfp16arith-u16.c &

################################# x86 128-bit #################################
tools/xngen src/f16-vunary/sse2.c.in -D OP=ABS -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vabs-sse2-u8.c &
tools/xngen src/f16-vunary/sse2.c.in -D OP=ABS -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vabs-sse2-u16.c &
tools/xngen src/f16-vunary/sse2.c.in -D OP=NEG -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vneg-sse2-u8.c &
tools/xngen src/f16-vunary/sse2.c.in -D OP=NEG -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vneg-sse2-u16.c &
tools/xngen src/f16-vunary/f16c.c.in -D OP=SQR -D BATCH_TILE=8  -o src/f16-vunary/gen/f16-vsqr-f16c-u8.c &
tools/xngen src/f16-vunary/f16c.c.in -D OP=SQR -D BATCH_TILE=16 -o src/f16-vunary/gen/f16-vsqr-f16c-u16.c &

wait
