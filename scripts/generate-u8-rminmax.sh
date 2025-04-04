#!/bin/sh
# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=U8 -D ACCUMULATORS=1 -D OP=MAX -D ARCH=neon -o src/u8-rminmax/gen/u8-rmax-neon-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=neon -o src/u8-rminmax/gen/u8-rmax-neon-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=U8 -D ACCUMULATORS=3 -D OP=MAX -D ARCH=neon -o src/u8-rminmax/gen/u8-rmax-neon-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=neon -o src/u8-rminmax/gen/u8-rmax-neon-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=4 -D OP=MAX -D ARCH=neon -o src/u8-rminmax/gen/u8-rmax-neon-u64-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=U8 -D ACCUMULATORS=1 -D OP=MIN -D ARCH=neon -o src/u8-rminmax/gen/u8-rmin-neon-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=neon -o src/u8-rminmax/gen/u8-rmin-neon-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=U8 -D ACCUMULATORS=3 -D OP=MIN -D ARCH=neon -o src/u8-rminmax/gen/u8-rmin-neon-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=neon -o src/u8-rminmax/gen/u8-rmin-neon-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=4 -D OP=MIN -D ARCH=neon -o src/u8-rminmax/gen/u8-rmin-neon-u64-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=U8 -D ACCUMULATORS=1 -D OP=MINMAX -D ARCH=neon -o src/u8-rminmax/gen/u8-rminmax-neon-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=neon -o src/u8-rminmax/gen/u8-rminmax-neon-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=U8 -D ACCUMULATORS=3 -D OP=MINMAX -D ARCH=neon -o src/u8-rminmax/gen/u8-rminmax-neon-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=neon -o src/u8-rminmax/gen/u8-rminmax-neon-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=4 -D OP=MINMAX -D ARCH=neon -o src/u8-rminmax/gen/u8-rminmax-neon-u64-acc4.c &

################################### x86 SSE2 ################################
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=U8 -D ACCUMULATORS=1 -D OP=MAX -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rmax-sse2-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rmax-sse2-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=U8 -D ACCUMULATORS=3 -D OP=MAX -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rmax-sse2-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rmax-sse2-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=4 -D OP=MAX -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rmax-sse2-u64-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=U8 -D ACCUMULATORS=1 -D OP=MIN -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rmin-sse2-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rmin-sse2-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=U8 -D ACCUMULATORS=3 -D OP=MIN -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rmin-sse2-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rmin-sse2-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=4 -D OP=MIN -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rmin-sse2-u64-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=16 -D DATATYPE=U8 -D ACCUMULATORS=1 -D OP=MINMAX -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rminmax-sse2-u16.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rminmax-sse2-u32-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=48 -D DATATYPE=U8 -D ACCUMULATORS=3 -D OP=MINMAX -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rminmax-sse2-u48-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rminmax-sse2-u64-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=64 -D DATATYPE=U8 -D ACCUMULATORS=4 -D OP=MINMAX -D ARCH=sse2 -o src/u8-rminmax/gen/u8-rminmax-sse2-u64-acc4.c &

################################## Wasm SIMD ##################################
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=wasmsimd -o src/u8-rminmax/gen/u8-rmax-wasmsimd-u32-acc2.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=wasmsimd -o src/u8-rminmax/gen/u8-rmin-wasmsimd-u32-acc2.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=32 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=wasmsimd -o src/u8-rminmax/gen/u8-rminmax-wasmsimd-u32-acc2.c &

#################################### Scalar ###################################
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=1 -D DATATYPE=U8 -D ACCUMULATORS=1 -D OP=MINMAX -D ARCH=scalar -o src/u8-rminmax/gen/u8-rminmax-scalar-u1.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=2 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=scalar -o src/u8-rminmax/gen/u8-rminmax-scalar-u2-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=3 -D DATATYPE=U8 -D ACCUMULATORS=3 -D OP=MINMAX -D ARCH=scalar -o src/u8-rminmax/gen/u8-rminmax-scalar-u3-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MINMAX -D ARCH=scalar -o src/u8-rminmax/gen/u8-rminmax-scalar-u4-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=U8 -D ACCUMULATORS=4 -D OP=MINMAX -D ARCH=scalar -o src/u8-rminmax/gen/u8-rminmax-scalar-u4-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=1 -D DATATYPE=U8 -D ACCUMULATORS=1 -D OP=MAX -D ARCH=scalar -o src/u8-rminmax/gen/u8-rmax-scalar-u1.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=2 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=scalar -o src/u8-rminmax/gen/u8-rmax-scalar-u2-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=3 -D DATATYPE=U8 -D ACCUMULATORS=3 -D OP=MAX -D ARCH=scalar -o src/u8-rminmax/gen/u8-rmax-scalar-u3-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MAX -D ARCH=scalar -o src/u8-rminmax/gen/u8-rmax-scalar-u4-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=U8 -D ACCUMULATORS=4 -D OP=MAX -D ARCH=scalar -o src/u8-rminmax/gen/u8-rmax-scalar-u4-acc4.c &

tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=1 -D DATATYPE=U8 -D ACCUMULATORS=1 -D OP=MIN -D ARCH=scalar -o src/u8-rminmax/gen/u8-rmin-scalar-u1.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=2 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=scalar -o src/u8-rminmax/gen/u8-rmin-scalar-u2-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=3 -D DATATYPE=U8 -D ACCUMULATORS=3 -D OP=MIN -D ARCH=scalar -o src/u8-rminmax/gen/u8-rmin-scalar-u3-acc3.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=U8 -D ACCUMULATORS=2 -D OP=MIN -D ARCH=scalar -o src/u8-rminmax/gen/u8-rmin-scalar-u4-acc2.c &
tools/xngen src/s8-rminmax/simd.c.in -D BATCH_TILE=4 -D DATATYPE=U8 -D ACCUMULATORS=4 -D OP=MIN -D ARCH=scalar -o src/u8-rminmax/gen/u8-rmin-scalar-u4-acc4.c &

wait
