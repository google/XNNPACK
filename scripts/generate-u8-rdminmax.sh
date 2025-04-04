#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=neon -o src/u8-rdminmax/gen/u8-rdmax-2p2x-neon-c32.c &

tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=neon -o src/u8-rdminmax/gen/u8-rdmin-2p2x-neon-c32.c &

################################### x86 SSE2 ################################
tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=sse2 -o src/u8-rdminmax/gen/u8-rdmax-2p2x-sse2-c32.c &

tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=sse2 -o src/u8-rdminmax/gen/u8-rdmin-2p2x-sse2-c32.c &

################################## Wasm SIMD ##################################
tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=wasmsimd -o src/u8-rdminmax/gen/u8-rdmax-2p2x-wasmsimd-c32.c &

tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=wasmsimd -o src/u8-rdminmax/gen/u8-rdmin-2p2x-wasmsimd-c32.c &

#################################### Scalar ###################################
tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=2 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=scalar -o src/u8-rdminmax/gen/u8-rdmax-2p2x-scalar-c2.c &

tools/xngen src/s8-rdminmax/simd.c.in -D CHANNELS=2 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=scalar -o src/u8-rdminmax/gen/u8-rdmin-2p2x-scalar-c2.c &

wait
