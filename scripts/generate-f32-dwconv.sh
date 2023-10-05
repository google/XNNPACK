#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-3p1c-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=3  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-3p1c-scalar-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-3p2c-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=3  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-3p2c-scalar-acc2.c &

tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-3p1c-minmax-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=3  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-3p1c-minmax-scalar-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-3p2c-minmax-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=3  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-3p2c-minmax-scalar-acc2.c &

tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-4p1c-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-4p1c-scalar-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-4p2c-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-4p2c-scalar-acc2.c &

tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-4p1c-minmax-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-4p1c-minmax-scalar-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-4p2c-minmax-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-4p2c-minmax-scalar-acc2.c &

tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-9p1c-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-9p1c-scalar-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-9p2c-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-9p2c-scalar-acc2.c &

tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-9p1c-minmax-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-9p1c-minmax-scalar-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-9p2c-minmax-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-9p2c-minmax-scalar-acc2.c &

tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-25p1c-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-25p1c-scalar-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-25p2c-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-25p2c-scalar-acc2.c &

tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-25p1c-minmax-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-25p1c-minmax-scalar-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-25p2c-minmax-scalar.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-25p2c-minmax-scalar-acc2.c &

tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=2 -D MIDDLE_PASS_TILE=2 -D LAST_PASS_TILE=2 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-2f2m2l1c1s1r-scalar.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=2 -D MIDDLE_PASS_TILE=2 -D LAST_PASS_TILE=2 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-2f2m2l1c1s1r-scalar-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-3f3m3l1c1s1r-scalar.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-3f3m3l1c1s1r-scalar-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=2 -D MIDDLE_PASS_TILE=2 -D LAST_PASS_TILE=2 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-2f2m2l4c1s1r-scalar.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=2 -D MIDDLE_PASS_TILE=2 -D LAST_PASS_TILE=2 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-2f2m2l4c1s1r-scalar-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-5f5m5l1c1s1r-scalar.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-5f5m5l1c1s1r-scalar-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-6f6m7l1c1s1r-scalar.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-6f6m7l1c1s1r-scalar-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-8f8m9l1c1s1r-scalar.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/f32-dwconv-8f8m9l1c1s1r-scalar-acc2.c &

tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=2 -D MIDDLE_PASS_TILE=2 -D LAST_PASS_TILE=2 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-2f2m2l1c1s1r-minmax-scalar.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=2 -D MIDDLE_PASS_TILE=2 -D LAST_PASS_TILE=2 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-2f2m2l1c1s1r-minmax-scalar-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=2 -D MIDDLE_PASS_TILE=2 -D LAST_PASS_TILE=2 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-2f2m2l4c1s1r-minmax-scalar.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=4 -D FIRST_PASS_TILE=2 -D MIDDLE_PASS_TILE=2 -D LAST_PASS_TILE=2 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-2f2m2l4c1s1r-minmax-scalar-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-5f5m5l1c1s1r-minmax-scalar.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-5f5m5l1c1s1r-minmax-scalar-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-6f6m7l1c1s1r-minmax-scalar.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-6f6m7l1c1s1r-minmax-scalar-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-8f8m9l1c1s1r-minmax-scalar.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-8f8m9l1c1s1r-minmax-scalar-acc2.c &

### WAsm-specific micro-kernels
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-3p1c-minmax-wasm.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=3  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-3p1c-minmax-wasm-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-3p2c-minmax-wasm.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=3  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-3p2c-minmax-wasm-acc2.c &

tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-4p1c-minmax-wasm.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-4p1c-minmax-wasm-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-4p2c-minmax-wasm.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-4p2c-minmax-wasm-acc2.c &

tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-9p1c-minmax-wasm.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-9p1c-minmax-wasm-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-9p2c-minmax-wasm.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-9p2c-minmax-wasm-acc2.c &

tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-25p1c-minmax-wasm.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-25p1c-minmax-wasm-acc2.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-25p2c-minmax-wasm.c &
tools/xngen src/f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-25p2c-minmax-wasm-acc2.c &

tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-3f3m3l1c1s1r-minmax-wasm.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-3f3m3l1c1s1r-minmax-wasm-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-5f5m5l1c1s1r-minmax-wasm.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-5f5m5l1c1s1r-minmax-wasm-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-6f6m7l1c1s1r-minmax-wasm.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-6f6m7l1c1s1r-minmax-wasm-acc2.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-8f8m9l1c1s1r-minmax-wasm.c &
tools/xngen src/f32-dwconv/multipass-scalar.c.in -D CHANNEL_TILE=1 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/f32-dwconv-8f8m9l1c1s1r-minmax-wasm-acc2.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmsimd-arm.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmsimd-arm-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmsimd-arm.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmsimd-arm-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmsimd-x86.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmsimd-x86-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmsimd-x86.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmsimd-x86-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmrelaxedsimd-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmrelaxedsimd-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-wasmrelaxedsimd-fma-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-wasmrelaxedsimd-fma-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmsimd-arm.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmsimd-arm-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmsimd-arm.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmsimd-arm-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmsimd-x86.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmsimd-x86-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmsimd-x86.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmsimd-x86-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmrelaxedsimd-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmrelaxedsimd-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-wasmrelaxedsimd-fma-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-wasmrelaxedsimd-fma-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-3p4c-wasmsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-4p4c-wasmsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-3p8c-wasmsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-4p8c-wasmsimd.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-3p4c-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-4p4c-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-3p8c-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-4p8c-wasmrelaxedsimd-fma.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmsimd-arm.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmsimd-arm-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmsimd-arm.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmsimd-arm-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmsimd-x86.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmsimd-x86-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmsimd-x86.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmsimd-x86-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmrelaxedsimd-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmrelaxedsimd-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-wasmrelaxedsimd-fma-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-wasmrelaxedsimd-fma-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-9p4c-wasmsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-9p4c-wasmsimd-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-9p8c-wasmsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-9p8c-wasmsimd-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-9p4c-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-9p8c-wasmrelaxedsimd-fma.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmsimd-arm.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmsimd-arm-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmsimd-arm.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmsimd-arm-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmsimd-x86.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmsimd-x86-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmsimd-x86.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmsimd-x86-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmrelaxedsimd-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmrelaxedsimd-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-wasmrelaxedsimd-fma-acc2.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-wasmrelaxedsimd-fma-acc2.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-25p4c-wasmsimd.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-25p8c-wasmsimd.c &

tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-25p4c-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/unipass-wasmsimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-25p8c-wasmrelaxedsimd-fma.c &

tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-3f3m3l4c4s4r-wasmsimd.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-3f3m3l4c4s4r-wasmsimd-acc2.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-3f3m3l8c4s4r-wasmsimd.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-3f3m3l8c4s4r-wasmsimd-acc2.c &

tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-wasmsimd.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=LINEAR                 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-wasmsimd-acc2.c &

tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -D FMA=1 -D ACTIVATION=LINEAR -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-wasmrelaxedsimd-fma-acc2.c &

tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-3f3m3l4c4s4r-minmax-wasmsimd-arm.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-3f3m3l4c4s4r-minmax-wasmsimd-arm-acc2.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-3f3m3l8c4s4r-minmax-wasmsimd-arm.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-3f3m3l8c4s4r-minmax-wasmsimd-arm-acc2.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmsimd-arm.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=ARM     -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmsimd-arm-acc2.c &

tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-3f3m3l4c4s4r-minmax-wasmsimd-x86.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-3f3m3l4c4s4r-minmax-wasmsimd-x86-acc2.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-3f3m3l8c4s4r-minmax-wasmsimd-x86.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=3 -D MIDDLE_PASS_TILE=3 -D LAST_PASS_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-3f3m3l8c4s4r-minmax-wasmsimd-x86-acc2.c &

tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmsimd-x86.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=X86     -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmsimd-x86-acc2.c &

tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmrelaxedsimd.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -D FMA=0 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmrelaxedsimd-acc2.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmrelaxedsimd-fma.c &
tools/xngen src/f32-dwconv/multipass-wasmsimd.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -D FMA=1 -D ACTIVATION=MINMAX -D ARCH=RELAXED -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-wasmrelaxedsimd-fma-acc2.c &

################################### ARM NEON ##################################
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-neon-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-neon-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-neon-acc2.c &

tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-neonfma-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-neonfma-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-neonfma-acc2.c &

tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-neon-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-neon-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-neon-acc2.c &

tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-neonfma-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-neonfma-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-neonfma-acc2.c &

tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-neon-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-neon-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-neon-acc2.c &

tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-neonfma-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-neonfma-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-neonfma-acc2.c &

tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-neon-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-neon-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-neon.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-neon-acc2.c &

tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=4  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-neonfma-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-neonfma-acc2.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-neonfma.c &
tools/xngen src/f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-neonfma-acc2.c &

tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-neon.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-neon-acc2.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l8c4s4r-minmax-neon.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l8c4s4r-minmax-neon-acc2.c &

tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l4c4s4r-minmax-neon.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l4c4s4r-minmax-neon-acc2.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l8c4s4r-minmax-neon.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l8c4s4r-minmax-neon-acc2.c &

tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l4c4s4r-minmax-neon.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l4c4s4r-minmax-neon-acc2.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l8c4s4r-minmax-neon.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l8c4s4r-minmax-neon-acc2.c &

tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=1 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-neonfma.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=1 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-neonfma-acc2.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=1 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l8c4s4r-minmax-neonfma.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=1 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l8c4s4r-minmax-neonfma-acc2.c &

tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=1 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l4c4s4r-minmax-neonfma.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=1 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l4c4s4r-minmax-neonfma-acc2.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=1 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l8c4s4r-minmax-neonfma.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=1 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l8c4s4r-minmax-neonfma-acc2.c &

tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=1 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l4c4s4r-minmax-neonfma.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=1 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l4c4s4r-minmax-neonfma-acc2.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=1 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l8c4s4r-minmax-neonfma.c &
tools/xngen src/f32-dwconv/multipass-neon.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=1 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l8c4s4r-minmax-neonfma-acc2.c &

################################### x86 SSE ###################################
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-sse.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-3p4c-minmax-sse-acc2.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-sse.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-sse-acc2.c &

tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-sse.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-4p4c-minmax-sse-acc2.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-sse.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-sse-acc2.c &

tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-sse.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-9p4c-minmax-sse-acc2.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-sse.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-sse-acc2.c &

tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-sse.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-25p4c-minmax-sse-acc2.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-sse.c &
tools/xngen src/f32-dwconv/unipass-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-sse-acc2.c &

tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-sse.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l4c4s4r-minmax-sse-acc2.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l8c4s4r-minmax-sse.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l8c4s4r-minmax-sse-acc2.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l16c4s4r-minmax-sse.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l16c4s4r-minmax-sse-acc2.c &

tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l4c4s4r-minmax-sse.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l4c4s4r-minmax-sse-acc2.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l8c4s4r-minmax-sse.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l8c4s4r-minmax-sse-acc2.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l16c4s4r-minmax-sse.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l16c4s4r-minmax-sse-acc2.c &

tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l4c4s4r-minmax-sse.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=4  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l4c4s4r-minmax-sse-acc2.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l8c4s4r-minmax-sse.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l8c4s4r-minmax-sse-acc2.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l16c4s4r-minmax-sse.c &
tools/xngen src/f32-dwconv/multipass-sse.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l16c4s4r-minmax-sse-acc2.c &

################################### x86 AVX ###################################
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-avx.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-avx-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-avx.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-avx-acc2.c &

tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-avx.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-avx-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-avx.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-avx-acc2.c &

tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-avx.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-avx-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-avx.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-avx-acc2.c &

tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-avx.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-avx-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-avx.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-avx-acc2.c &

tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l8c8s4r-minmax-avx.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l8c8s4r-minmax-avx-acc2.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l16c8s4r-minmax-avx.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l16c8s4r-minmax-avx-acc2.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l8c8s4r-minmax-avx.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l8c8s4r-minmax-avx-acc2.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l16c8s4r-minmax-avx.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=6 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=7 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-6f6m7l16c8s4r-minmax-avx-acc2.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l8c8s4r-minmax-avx.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l8c8s4r-minmax-avx-acc2.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l16c8s4r-minmax-avx.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=8 -D MIDDLE_PASS_TILE=8 -D LAST_PASS_TILE=9 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-8f8m9l16c8s4r-minmax-avx-acc2.c &

tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-fma3.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=3 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-3p8c-minmax-fma3-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-fma3.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-fma3-acc2.c &

tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-fma3.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-4p8c-minmax-fma3-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-fma3.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-fma3-acc2.c &

tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-fma3.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-9p8c-minmax-fma3-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-fma3.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-fma3-acc2.c &

tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-fma3.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-25p8c-minmax-fma3-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-fma3.c &
tools/xngen src/f32-dwconv/unipass-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-fma3-acc2.c &

tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l8c8s4r-minmax-fma3.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l8c8s4r-minmax-fma3-acc2.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l16c8s4r-minmax-fma3.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l16c8s4r-minmax-fma3-acc2.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l32c8s4r-minmax-fma3.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l32c8s4r-minmax-fma3-acc2.c &

tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=7 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=6 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-7f6m6l8c8s4r-minmax-fma3.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=8  -D FIRST_PASS_TILE=7 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=6 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-7f6m6l8c8s4r-minmax-fma3-acc2.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=7 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=6 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-7f6m6l16c8s4r-minmax-fma3.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=7 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=6 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-7f6m6l16c8s4r-minmax-fma3-acc2.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=7 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=6 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-7f6m6l32c8s4r-minmax-fma3.c &
tools/xngen src/f32-dwconv/multipass-avx.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=7 -D MIDDLE_PASS_TILE=6 -D LAST_PASS_TILE=6 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-7f6m6l32c8s4r-minmax-fma3-acc2.c &

################################# x86 AVX-512 #################################
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-avx512f.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-3p16c-minmax-avx512f-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-3p32c-minmax-avx512f.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-3p32c-minmax-avx512f-acc2.c &

tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-avx512f.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-4p16c-minmax-avx512f-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-4p32c-minmax-avx512f.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-4p32c-minmax-avx512f-acc2.c &

tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-avx512f.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-9p16c-minmax-avx512f-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-9p32c-minmax-avx512f.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-9p32c-minmax-avx512f-acc2.c &

tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-avx512f.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-25p16c-minmax-avx512f-acc2.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-25p32c-minmax-avx512f.c &
tools/xngen src/f32-dwconv/unipass-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-25p32c-minmax-avx512f-acc2.c &

tools/xngen src/f32-dwconv/multipass-avx512.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l16c16s1r-minmax-avx512f.c &
tools/xngen src/f32-dwconv/multipass-avx512.c.in -D CHANNEL_TILE=16 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l16c16s1r-minmax-avx512f-acc2.c &
tools/xngen src/f32-dwconv/multipass-avx512.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l32c16s1r-minmax-avx512f.c &
tools/xngen src/f32-dwconv/multipass-avx512.c.in -D CHANNEL_TILE=32 -D FIRST_PASS_TILE=5 -D MIDDLE_PASS_TILE=5 -D LAST_PASS_TILE=5 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/f32-dwconv-5f5m5l32c16s1r-minmax-avx512f-acc2.c &

wait
