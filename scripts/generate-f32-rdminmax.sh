#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=neon -o src/f32-rdminmax/gen/f32-rdmax-2p2x-neon-c32.c &

tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=neon -o src/f32-rdminmax/gen/f32-rdmin-2p2x-neon-c32.c &

################################### x86 SSE2 ##################################
tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=sse2 -o src/f32-rdminmax/gen/f32-rdmax-2p2x-sse2-c32.c &

tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=sse2 -o src/f32-rdminmax/gen/f32-rdmin-2p2x-sse2-c32.c &

################################### x86 AVX ###################################
tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=avx -o src/f32-rdminmax/gen/f32-rdmax-2p2x-avx-c32.c &

tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=avx -o src/f32-rdminmax/gen/f32-rdmin-2p2x-avx-c32.c &

################################### x86 AVX512 ################################
tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=avx512f -o src/f32-rdminmax/gen/f32-rdmax-2p2x-avx512f-c32.c &

tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=avx512f -o src/f32-rdminmax/gen/f32-rdmin-2p2x-avx512f-c32.c &

################################## Wasm SIMD ##################################
tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=wasmsimd -o src/f32-rdminmax/gen/f32-rdmax-2p2x-wasmsimd-c32.c &

tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=wasmsimd -o src/f32-rdminmax/gen/f32-rdmin-2p2x-wasmsimd-c32.c &

################################## Hexagon HVX ################################
tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=hvx -o src/f32-rdminmax/gen/f32-rdmax-2p2x-hvx-c32.c &

tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=32 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=hvx -o src/f32-rdminmax/gen/f32-rdmin-2p2x-hvx-c32.c &

#################################### Scalar ###################################
tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=2 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=scalar -o src/f32-rdminmax/gen/f32-rdmax-2p2x-scalar-c2.c &

tools/xngen src/f32-rdminmax/simd.c.in -D CHANNELS=2 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=scalar -o src/f32-rdminmax/gen/f32-rdmin-2p2x-scalar-c2.c &

wait
