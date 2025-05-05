#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=neonfp16arith -o src/f16-rdminmax/gen/f16-rdmax-2p2x-neonfp16arith-c32.c &

tools/xngen src/f16-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=neonfp16arith -o src/f16-rdminmax/gen/f16-rdmin-2p2x-neonfp16arith-c32.c &

#################################### Scalar ###################################
tools/xngen src/f16-rdminmax/simd.c.in -D CHANNELS=2 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=scalar -o src/f16-rdminmax/gen/f16-rdmax-2p2x-scalar-c2.c &

tools/xngen src/f16-rdminmax/simd.c.in -D CHANNELS=2 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=scalar -o src/f16-rdminmax/gen/f16-rdmin-2p2x-scalar-c2.c &

wait
