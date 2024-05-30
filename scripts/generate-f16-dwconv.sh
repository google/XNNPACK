#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p8c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p8c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p16c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p16c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p32c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p32c-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p8c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p8c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p16c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p16c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p32c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p32c-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p8c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p8c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p16c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p16c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p32c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p32c-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p8c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p8c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p16c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p16c-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p32c-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/unipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p32c-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l8c8s4r-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l8c8s4r-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l16c8s4r-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l16c8s4r-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l32c8s4r-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l32c8s4r-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l8c8s4r-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l8c8s4r-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l16c8s4r-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l16c8s4r-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l32c8s4r-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l32c8s4r-minmax-neonfp16arith-acc2.c &

tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l8c8s4r-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l8c8s4r-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l16c8s4r-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l16c8s4r-minmax-neonfp16arith-acc2.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l32c8s4r-minmax-neonfp16arith.c &
tools/xngen src/f16-dwconv/multipass-neonfp16arith.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l32c8s4r-minmax-neonfp16arith-acc2.c &

################################### x86 FMA3 ##################################
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p8c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p8c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p16c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p16c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-3p32c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-3p32c-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p8c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p8c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p16c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p16c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-4p32c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-4p32c-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p8c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p8c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p16c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p16c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-9p32c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-9p32c-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p8c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p8c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p16c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p16c-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-25p32c-minmax-fma3.c &
tools/xngen src/f16-dwconv/unipass-fma3.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-25p32c-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l8c8s4r-minmax-fma3.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l8c8s4r-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l16c8s4r-minmax-fma3.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l16c8s4r-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l32c8s4r-minmax-fma3.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-5f5m5l32c8s4r-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l8c8s4r-minmax-fma3.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l8c8s4r-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l16c8s4r-minmax-fma3.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l16c8s4r-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l32c8s4r-minmax-fma3.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-6f6m7l32c8s4r-minmax-fma3-acc2.c &

tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l8c8s4r-minmax-fma3.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l8c8s4r-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l16c8s4r-minmax-fma3.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l16c8s4r-minmax-fma3-acc2.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l32c8s4r-minmax-fma3.c &
tools/xngen src/f16-dwconv/multipass-fma3.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -o src/f16-dwconv/gen/f16-dwconv-8f8m9l32c8s4r-minmax-fma3-acc2.c &

wait
