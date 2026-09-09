#!/bin/sh
# Copyright 2020 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-ibilinear-chw/scalar.c.in -D PIXEL_TILE=1 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-scalar-p1.c &
tools/xngen src/f32-ibilinear-chw/scalar.c.in -D PIXEL_TILE=2 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-scalar-p2.c &
tools/xngen src/f32-ibilinear-chw/scalar.c.in -D PIXEL_TILE=4 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-scalar-p4.c &

############################### WebAssembly SIMD ##############################
tools/xngen src/f32-ibilinear-chw/wasmsimd.c.in -D PIXEL_TILE=4 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-wasmsimd-p4.c &
tools/xngen src/f32-ibilinear-chw/wasmsimd.c.in -D PIXEL_TILE=8 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-wasmsimd-p8.c &

############################### ARM NEON ##############################
tools/xngen src/f32-ibilinear-chw/neon.c.in -D PIXEL_TILE=4  -D FMA=0 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-neon-p4.c &
tools/xngen src/f32-ibilinear-chw/neon.c.in -D PIXEL_TILE=8  -D FMA=0 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-neon-p8.c &
tools/xngen src/f32-ibilinear-chw/neon.c.in -D PIXEL_TILE=16 -D FMA=0 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-neon-p16.c &

tools/xngen src/f32-ibilinear-chw/neon.c.in -D PIXEL_TILE=4  -D FMA=1 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-neonfma-p4.c &
tools/xngen src/f32-ibilinear-chw/neon.c.in -D PIXEL_TILE=8  -D FMA=1 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-neonfma-p8.c &
tools/xngen src/f32-ibilinear-chw/neon.c.in -D PIXEL_TILE=16 -D FMA=1 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-neonfma-p16.c &

################################### x86 SSE ###################################
tools/xngen src/f32-ibilinear-chw/sse.c.in -D PIXEL_TILE=4 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-sse-p4.c &
tools/xngen src/f32-ibilinear-chw/sse.c.in -D PIXEL_TILE=8 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-sse-p8.c &

################################## RISC-V RVV #################################
tools/xngen src/f32-ibilinear-chw/rvv.c.in -D CHANNEL_TILE=1 LMUL=1 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-rvv-1x1v.c &
tools/xngen src/f32-ibilinear-chw/rvv.c.in -D CHANNEL_TILE=1 LMUL=2 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-rvv-1x2v.c &
tools/xngen src/f32-ibilinear-chw/rvv.c.in -D CHANNEL_TILE=2 LMUL=1 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-rvv-2x1v.c &
tools/xngen src/f32-ibilinear-chw/rvv.c.in -D CHANNEL_TILE=2 LMUL=2 -o src/f32-ibilinear-chw/gen/f32-ibilinear-chw-rvv-2x2v.c &

wait
