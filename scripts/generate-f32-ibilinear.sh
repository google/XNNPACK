#!/bin/sh
# Copyright 2019 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-ibilinear/scalar.c.in -D CHANNEL_TILE=1 -D PIXEL_TILE=1 -o src/f32-ibilinear/gen/f32-ibilinear-scalar-c1.c &
tools/xngen src/f32-ibilinear/scalar.c.in -D CHANNEL_TILE=2 -D PIXEL_TILE=1 -o src/f32-ibilinear/gen/f32-ibilinear-scalar-c2.c &
tools/xngen src/f32-ibilinear/scalar.c.in -D CHANNEL_TILE=4 -D PIXEL_TILE=1 -o src/f32-ibilinear/gen/f32-ibilinear-scalar-c4.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-ibilinear/wasmsimd.c.in -D CHANNEL_TILE=4 -D PIXEL_TILE=1 -D FMA=0 -o src/f32-ibilinear/gen/f32-ibilinear-wasmsimd-c4.c &
tools/xngen src/f32-ibilinear/wasmsimd.c.in -D CHANNEL_TILE=8 -D PIXEL_TILE=1 -D FMA=0 -o src/f32-ibilinear/gen/f32-ibilinear-wasmsimd-c8.c &

tools/xngen src/f32-ibilinear/wasmsimd.c.in -D CHANNEL_TILE=4 -D PIXEL_TILE=1 -D FMA=1 -o src/f32-ibilinear/gen/f32-ibilinear-wasmrelaxedsimd-c4.c &
tools/xngen src/f32-ibilinear/wasmsimd.c.in -D CHANNEL_TILE=8 -D PIXEL_TILE=1 -D FMA=1 -o src/f32-ibilinear/gen/f32-ibilinear-wasmrelaxedsimd-c8.c &

################################### ARM NEON ##################################
tools/xngen src/f32-ibilinear/neon.c.in -D CHANNEL_TILE=4 -D PIXEL_TILE=1 -D FMA=0 -o src/f32-ibilinear/gen/f32-ibilinear-neon-c4.c &
tools/xngen src/f32-ibilinear/neon.c.in -D CHANNEL_TILE=8 -D PIXEL_TILE=1 -D FMA=0 -o src/f32-ibilinear/gen/f32-ibilinear-neon-c8.c &

tools/xngen src/f32-ibilinear/neon.c.in -D CHANNEL_TILE=4 -D PIXEL_TILE=1 -D FMA=1 -o src/f32-ibilinear/gen/f32-ibilinear-neonfma-c4.c &
tools/xngen src/f32-ibilinear/neon.c.in -D CHANNEL_TILE=8 -D PIXEL_TILE=1 -D FMA=1 -o src/f32-ibilinear/gen/f32-ibilinear-neonfma-c8.c &

################################### x86 SSE ###################################
tools/xngen src/f32-ibilinear/sse.c.in -D CHANNEL_TILE=4 -D PIXEL_TILE=1 -o src/f32-ibilinear/gen/f32-ibilinear-sse-c4.c &
tools/xngen src/f32-ibilinear/sse.c.in -D CHANNEL_TILE=8 -D PIXEL_TILE=1 -o src/f32-ibilinear/gen/f32-ibilinear-sse-c8.c &

wait
