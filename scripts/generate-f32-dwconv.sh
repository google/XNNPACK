#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x4-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x4-scalar-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x4-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x4-scalar-acc2.c

tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x4-minmax-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x4-minmax-scalar-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x4-minmax-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x4-minmax-scalar-acc2.c

tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x9-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x9-scalar-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x9-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x9-scalar-acc2.c

tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x9-minmax-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x9-minmax-scalar-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x9-minmax-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x9-minmax-scalar-acc2.c

tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x25-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x25-scalar-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x25-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x25-scalar-acc2.c

tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x25-minmax-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x25-minmax-scalar-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x25-minmax-scalar.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=0 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x25-minmax-scalar-acc2.c

### WAsm-specific micro-kernels
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x4-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x4-wasm-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x4-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x4-wasm-acc2.c

tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x4-minmax-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x4-minmax-wasm-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x4-minmax-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x4-minmax-wasm-acc2.c

tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x9-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x9-wasm-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x9-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x9-wasm-acc2.c

tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x9-minmax-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x9-minmax-wasm-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x9-minmax-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x9-minmax-wasm-acc2.c

tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x25-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up1x25-wasm-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x25-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=LINEAR -o src/f32-dwconv/gen/up2x25-wasm-acc2.c

tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x25-minmax-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=1 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up1x25-minmax-wasm-acc2.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x25-minmax-wasm.c
tools/xngen src/f32-dwconv/up-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D WASM=1 -D ACTIVATION=MINMAX -o src/f32-dwconv/gen/up2x25-minmax-wasm-acc2.c

################################### ARM NEON ##################################
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/up4x4-minmax-neon.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/up4x4-minmax-neon-acc2.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/up8x4-minmax-neon.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/up8x4-minmax-neon-acc2.c

tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/up4x4-minmax-neonfma.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/up4x4-minmax-neonfma-acc2.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/up8x4-minmax-neonfma.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/up8x4-minmax-neonfma-acc2.c

tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/up4x9-minmax-neon.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/up4x9-minmax-neon-acc2.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/up8x9-minmax-neon.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/up8x9-minmax-neon-acc2.c

tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/up4x9-minmax-neonfma.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/up4x9-minmax-neonfma-acc2.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/up8x9-minmax-neonfma.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/up8x9-minmax-neonfma-acc2.c

tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/up4x25-minmax-neon.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/up4x25-minmax-neon-acc2.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=0 -o src/f32-dwconv/gen/up8x25-minmax-neon.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=0 -o src/f32-dwconv/gen/up8x25-minmax-neon-acc2.c

tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/up4x25-minmax-neonfma.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/up4x25-minmax-neonfma-acc2.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -D FMA=1 -o src/f32-dwconv/gen/up8x25-minmax-neonfma.c
tools/xngen src/f32-dwconv/up-neon.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -D FMA=1 -o src/f32-dwconv/gen/up8x25-minmax-neonfma-acc2.c

################################### x86 SSE ###################################
tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up4x4-minmax-sse.c
tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up4x4-minmax-sse-acc2.c
tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x4-minmax-sse.c
tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x4-minmax-sse-acc2.c

tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up4x9-minmax-sse.c
tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up4x9-minmax-sse-acc2.c
tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x9-minmax-sse.c
tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x9-minmax-sse-acc2.c

tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up4x25-minmax-sse.c
tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up4x25-minmax-sse-acc2.c
tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x25-minmax-sse.c
tools/xngen src/f32-dwconv/up-sse.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x25-minmax-sse-acc2.c

################################### x86 AVX ###################################
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x4-minmax-avx.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x4-minmax-avx-acc2.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up16x4-minmax-avx.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up16x4-minmax-avx-acc2.c

tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x9-minmax-avx.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x9-minmax-avx-acc2.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up16x9-minmax-avx.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up16x9-minmax-avx-acc2.c

tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x25-minmax-avx.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x25-minmax-avx-acc2.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D FMA=0 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up16x25-minmax-avx.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D FMA=0 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up16x25-minmax-avx-acc2.c

tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x4-minmax-fma3.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x4-minmax-fma3-acc2.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up16x4-minmax-fma3.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up16x4-minmax-fma3-acc2.c

tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x9-minmax-fma3.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x9-minmax-fma3-acc2.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up16x9-minmax-fma3.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up16x9-minmax-fma3-acc2.c

tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x25-minmax-fma3.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x25-minmax-fma3-acc2.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D FMA=3 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up16x25-minmax-fma3.c
tools/xngen src/f32-dwconv/up-avx.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D FMA=3 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up16x25-minmax-fma3-acc2.c

################################# x86 AVX-512 #################################
tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up16x4-minmax-avx512f.c
tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up16x4-minmax-avx512f-acc2.c
tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up32x4-minmax-avx512f.c
tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up32x4-minmax-avx512f-acc2.c

tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up16x9-minmax-avx512f.c
tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up16x9-minmax-avx512f-acc2.c
tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up32x9-minmax-avx512f.c
tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up32x9-minmax-avx512f-acc2.c

tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up16x25-minmax-avx512f.c
tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up16x25-minmax-avx512f-acc2.c
tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up32x25-minmax-avx512f.c
tools/xngen src/f32-dwconv/up-avx512.c.in -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up32x25-minmax-avx512f-acc2.c

#################################### PSIMD ####################################
tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up4x4-minmax-psimd.c
tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up4x4-minmax-psimd-acc2.c
tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x4-minmax-psimd.c
tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=4 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x4-minmax-psimd-acc2.c

tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up4x9-minmax-psimd.c
tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up4x9-minmax-psimd-acc2.c
tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x9-minmax-psimd.c
tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=9 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x9-minmax-psimd-acc2.c

tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up4x25-minmax-psimd.c
tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=4 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up4x25-minmax-psimd-acc2.c
tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=1 -o src/f32-dwconv/gen/up8x25-minmax-psimd.c
tools/xngen src/f32-dwconv/up-psimd.c.in -D CHANNEL_TILE=8 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/f32-dwconv/gen/up8x25-minmax-psimd-acc2.c

################################## Unit tests #################################
tools/generate-dwconv-test.py --spec test/f32-dwconv.yaml --output test/f32-dwconv.cc
tools/generate-dwconv-test.py --spec test/f32-dwconv-minmax.yaml --output test/f32-dwconv-minmax.cc
