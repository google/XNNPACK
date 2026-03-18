#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=neon -o src/s8-rdminmax/gen/s8-rdmax-2p2x-neon-u32.c &

tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=neon -o src/s8-rdminmax/gen/s8-rdmin-2p2x-neon-u32.c &

################################### x86 SSE4.1 ################################
tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=sse41 -o src/s8-rdminmax/gen/s8-rdmax-2p2x-sse41-u32.c &

tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=sse41 -o src/s8-rdminmax/gen/s8-rdmin-2p2x-sse41-u32.c &

################################## Wasm SIMD ##################################
tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=wasmsimd -o src/s8-rdminmax/gen/s8-rdmax-2p2x-wasmsimd-u32.c &

tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=wasmsimd -o src/s8-rdminmax/gen/s8-rdmin-2p2x-wasmsimd-u32.c &

################################## Hexagon HVX ################################
tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=128 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=hvx -o src/s8-rdminmax/gen/s8-rdmax-2p2x-hvx-u128.c &

tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=128 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=hvx -o src/s8-rdminmax/gen/s8-rdmin-2p2x-hvx-u128.c &

#################################### Scalar ###################################
tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=2 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=scalar -o src/s8-rdminmax/gen/s8-rdmax-2p2x-scalar-u2.c &

tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=2 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=scalar -o src/s8-rdminmax/gen/s8-rdmin-2p2x-scalar-u2.c &

wait
