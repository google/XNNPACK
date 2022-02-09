#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


################################### ARM NEON ##################################
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up8x3-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up8x3-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up16x3-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up16x3-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up32x3-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up32x3-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up8x4-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up8x4-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up16x4-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up16x4-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up32x4-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up32x4-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up8x9-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up8x9-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up16x9-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up16x9-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up32x9-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up32x9-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up8x25-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up8x25-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up16x25-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up16x25-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up32x25-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/up-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up32x25-minmax-neonfp16arith-acc2.c &

################################### x86 FMA3 ##################################
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up8x3-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up8x3-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up16x3-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up16x3-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up32x3-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up32x3-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up8x4-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up8x4-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up16x4-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up16x4-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up32x4-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up32x4-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up8x9-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up8x9-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up16x9-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up16x9-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up32x9-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up32x9-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up8x25-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up8x25-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up16x25-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up16x25-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/up32x25-minmax-fma3.c &
tools/xngen src/f16-dwconv/up-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/up32x25-minmax-fma3-acc2.c &

################################## Unit tests #################################
tools/generate-dwconv-test.py --spec test/f16-dwconv-minmax.yaml --output test/f16-dwconv-minmax.cc &

wait
