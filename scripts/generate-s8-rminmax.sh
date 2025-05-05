#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MAX -D ARCH=neon -o src/s8-rminmax/gen/s8-rmax-neon-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=neon -o src/s8-rminmax/gen/s8-rmax-neon-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=S8 -D ACCUMULATORS=3 -D OP=MAX -D ARCH=neon -o src/s8-rminmax/gen/s8-rmax-neon-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=neon -o src/s8-rminmax/gen/s8-rmax-neon-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=4 -D OP=MAX -D ARCH=neon -o src/s8-rminmax/gen/s8-rmax-neon-u64-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MIN -D ARCH=neon -o src/s8-rminmax/gen/s8-rmin-neon-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=neon -o src/s8-rminmax/gen/s8-rmin-neon-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=S8 -D ACCUMULATORS=3 -D OP=MIN -D ARCH=neon -o src/s8-rminmax/gen/s8-rmin-neon-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=neon -o src/s8-rminmax/gen/s8-rmin-neon-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=4 -D OP=MIN -D ARCH=neon -o src/s8-rminmax/gen/s8-rmin-neon-u64-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MINMAX -D ARCH=neon -o src/s8-rminmax/gen/s8-rminmax-neon-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=neon -o src/s8-rminmax/gen/s8-rminmax-neon-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=S8 -D ACCUMULATORS=3 -D OP=MINMAX -D ARCH=neon -o src/s8-rminmax/gen/s8-rminmax-neon-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=neon -o src/s8-rminmax/gen/s8-rminmax-neon-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=4 -D OP=MINMAX -D ARCH=neon -o src/s8-rminmax/gen/s8-rminmax-neon-u64-acc4.c &

################################### x86 SSE4.1 ################################
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MAX -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rmax-sse41-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rmax-sse41-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=S8 -D ACCUMULATORS=3 -D OP=MAX -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rmax-sse41-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rmax-sse41-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=4 -D OP=MAX -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rmax-sse41-u64-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MIN -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rmin-sse41-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rmin-sse41-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=S8 -D ACCUMULATORS=3 -D OP=MIN -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rmin-sse41-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rmin-sse41-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=4 -D OP=MIN -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rmin-sse41-u64-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MINMAX -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rminmax-sse41-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rminmax-sse41-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=S8 -D ACCUMULATORS=3 -D OP=MINMAX -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rminmax-sse41-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rminmax-sse41-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=4 -D OP=MINMAX -D ARCH=sse41 -o src/s8-rminmax/gen/s8-rminmax-sse41-u64-acc4.c &

################################## Wasm SIMD ##################################
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MAX -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rmax-wasmsimd-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rmax-wasmsimd-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=S8 -D ACCUMULATORS=3 -D OP=MAX -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rmax-wasmsimd-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rmax-wasmsimd-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=4 -D OP=MAX -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rmax-wasmsimd-u64-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MIN -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rmin-wasmsimd-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rmin-wasmsimd-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=S8 -D ACCUMULATORS=3 -D OP=MIN -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rmin-wasmsimd-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rmin-wasmsimd-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8 -D ACCUMULATORS=4 -D OP=MIN -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rmin-wasmsimd-u64-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MINMAX -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rminmax-wasmsimd-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rminmax-wasmsimd-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=S8  -D ACCUMULATORS=3 -D OP=MINMAX -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rminmax-wasmsimd-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8  -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rminmax-wasmsimd-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=S8  -D ACCUMULATORS=4 -D OP=MINMAX -D ARCH=wasmsimd -o src/s8-rminmax/gen/s8-rminmax-wasmsimd-u64-acc4.c &

################################### HEXAGON HVX ################################
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=256 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX    -D ARCH=hvx -o src/s8-rminmax/gen/s8-rmax-hvx-u256-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=256 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN    -D ARCH=hvx -o src/s8-rminmax/gen/s8-rmin-hvx-u256-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=256 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=hvx -o src/s8-rminmax/gen/s8-rminmax-hvx-u256-acc2.c &

#################################### Scalar ###################################
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=1 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MINMAX -D ARCH=scalar -o src/s8-rminmax/gen/s8-rminmax-scalar-u1.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=2 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=scalar -o src/s8-rminmax/gen/s8-rminmax-scalar-u2-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=3 -D DATATYPE=S8 -D ACCUMULATORS=3 -D OP=MINMAX -D ARCH=scalar -o src/s8-rminmax/gen/s8-rminmax-scalar-u3-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=scalar -o src/s8-rminmax/gen/s8-rminmax-scalar-u4-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=S8 -D ACCUMULATORS=4 -D OP=MINMAX -D ARCH=scalar -o src/s8-rminmax/gen/s8-rminmax-scalar-u4-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=1 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MAX -D ARCH=scalar -o src/s8-rminmax/gen/s8-rmax-scalar-u1.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=2 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=scalar -o src/s8-rminmax/gen/s8-rmax-scalar-u2-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=3 -D DATATYPE=S8 -D ACCUMULATORS=3 -D OP=MAX -D ARCH=scalar -o src/s8-rminmax/gen/s8-rmax-scalar-u3-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=scalar -o src/s8-rminmax/gen/s8-rmax-scalar-u4-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=S8 -D ACCUMULATORS=4 -D OP=MAX -D ARCH=scalar -o src/s8-rminmax/gen/s8-rmax-scalar-u4-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=1 -D DATATYPE=S8 -D ACCUMULATORS=1 -D OP=MIN -D ARCH=scalar -o src/s8-rminmax/gen/s8-rmin-scalar-u1.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=2 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=scalar -o src/s8-rminmax/gen/s8-rmin-scalar-u2-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=3 -D DATATYPE=S8 -D ACCUMULATORS=3 -D OP=MIN -D ARCH=scalar -o src/s8-rminmax/gen/s8-rmin-scalar-u3-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=S8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=scalar -o src/s8-rminmax/gen/s8-rmin-scalar-u4-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=S8 -D ACCUMULATORS=4 -D OP=MIN -D ARCH=scalar -o src/s8-rminmax/gen/s8-rmin-scalar-u4-acc4.c &

wait
