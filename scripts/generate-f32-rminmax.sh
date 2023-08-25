#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEON ##################################
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-neon-u4.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-neon-u8-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-neon-u12-acc3.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-neon-u16-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-neon-u16-acc4.c &

tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-neon-u4.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-neon-u8-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-neon-u12-acc3.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-neon-u16-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-neon-u16-acc4.c &

tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-neon-u4.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-neon-u8-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-neon-u12-acc3.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-neon-u16-acc2.c &
tools/xngen src/f32-rminmax/neon.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-neon-u16-acc4.c &

################################### x86 SSE ###################################
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-sse-u4.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-sse-u8-acc2.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-sse-u12-acc3.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-sse-u16-acc2.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-sse-u16-acc4.c &

tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-sse-u4.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-sse-u8-acc2.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-sse-u12-acc3.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-sse-u16-acc2.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-sse-u16-acc4.c &

tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-sse-u4.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-sse-u8-acc2.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-sse-u12-acc3.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-sse-u16-acc2.c &
tools/xngen src/f32-rminmax/sse.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-sse-u16-acc4.c &

################################### x86 AVX ###################################
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-avx-u8.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-avx-u16-acc2.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-avx-u24-acc3.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-avx-u32-acc2.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-avx-u32-acc4.c &

tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-avx-u8.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-avx-u16-acc2.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-avx-u24-acc3.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-avx-u32-acc2.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-avx-u32-acc4.c &

tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-avx-u8.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-avx-u16-acc2.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-avx-u24-acc3.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-avx-u32-acc2.c &
tools/xngen src/f32-rminmax/avx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-avx-u32-acc4.c &

################################## x86 AVX512 #################################
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-avx512f-u16.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-avx512f-u32-acc2.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=48 -D ACCUMULATORS=3 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-avx512f-u48-acc3.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-avx512f-u64-acc2.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -D OP=MAX -o src/f32-rminmax/gen/f32-rmax-avx512f-u64-acc4.c &

tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-avx512f-u16.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-avx512f-u32-acc2.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=48 -D ACCUMULATORS=3 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-avx512f-u48-acc3.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-avx512f-u64-acc2.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -D OP=MIN -o src/f32-rminmax/gen/f32-rmin-avx512f-u64-acc4.c &

tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-avx512f-u16.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-avx512f-u32-acc2.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=48 -D ACCUMULATORS=3 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-avx512f-u48-acc3.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-avx512f-u64-acc2.c &
tools/xngen src/f32-rminmax/avx512f.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -D OP=MINMAX -o src/f32-rminmax/gen/f32-rminmax-avx512f-u64-acc4.c &

################################## Wasm SIMD ##################################
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-minmax-u4.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-minmax-u8-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-minmax-u12-acc3.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-minmax-u16-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-minmax-u16-acc4.c &

tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MIN -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-minmax-u4.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MIN -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-minmax-u8-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MIN -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-minmax-u12-acc3.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MIN -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-minmax-u16-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MIN -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-minmax-u16-acc4.c &

tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-pminmax-u4.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-pminmax-u8-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-pminmax-u12-acc3.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-pminmax-u16-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmax-wasmsimd-pminmax-u16-acc4.c &

tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MIN -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-pminmax-u4.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MIN -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-pminmax-u8-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MIN -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-pminmax-u12-acc3.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MIN -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-pminmax-u16-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MIN -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rmin-wasmsimd-pminmax-u16-acc4.c &

tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MINMAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rminmax-wasmsimd-minmax-u4.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MINMAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rminmax-wasmsimd-minmax-u8-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MINMAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rminmax-wasmsimd-minmax-u12-acc3.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MINMAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rminmax-wasmsimd-minmax-u16-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MINMAX -D MINMAX=MINMAX -o src/f32-rminmax/gen/f32-rminmax-wasmsimd-minmax-u16-acc4.c &

tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=4  -D ACCUMULATORS=1 -D OP=MINMAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rminmax-wasmsimd-pminmax-u4.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=8  -D ACCUMULATORS=2 -D OP=MINMAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rminmax-wasmsimd-pminmax-u8-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=12 -D ACCUMULATORS=3 -D OP=MINMAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rminmax-wasmsimd-pminmax-u12-acc3.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MINMAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rminmax-wasmsimd-pminmax-u16-acc2.c &
tools/xngen src/f32-rminmax/wasmsimd.c.in -D BATCH_TILE=16 -D ACCUMULATORS=4 -D OP=MINMAX -D MINMAX=PMINMAX -o src/f32-rminmax/gen/f32-rminmax-wasmsimd-pminmax-u16-acc4.c &

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MINMAX -D WASM=0 -o src/f32-rminmax/gen/f32-rminmax-scalar-u1.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MINMAX -D WASM=0 -o src/f32-rminmax/gen/f32-rminmax-scalar-u2-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MINMAX -D WASM=0 -o src/f32-rminmax/gen/f32-rminmax-scalar-u3-acc3.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MINMAX -D WASM=0 -o src/f32-rminmax/gen/f32-rminmax-scalar-u4-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MINMAX -D WASM=0 -o src/f32-rminmax/gen/f32-rminmax-scalar-u4-acc4.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MAX -D WASM=0 -o src/f32-rminmax/gen/f32-rmax-scalar-u1.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MAX -D WASM=0 -o src/f32-rminmax/gen/f32-rmax-scalar-u2-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MAX -D WASM=0 -o src/f32-rminmax/gen/f32-rmax-scalar-u3-acc3.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MAX -D WASM=0 -o src/f32-rminmax/gen/f32-rmax-scalar-u4-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MAX -D WASM=0 -o src/f32-rminmax/gen/f32-rmax-scalar-u4-acc4.c &

tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MIN -D WASM=0 -o src/f32-rminmax/gen/f32-rmin-scalar-u1.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MIN -D WASM=0 -o src/f32-rminmax/gen/f32-rmin-scalar-u2-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MIN -D WASM=0 -o src/f32-rminmax/gen/f32-rmin-scalar-u3-acc3.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MIN -D WASM=0 -o src/f32-rminmax/gen/f32-rmin-scalar-u4-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MIN -D WASM=0 -o src/f32-rminmax/gen/f32-rmin-scalar-u4-acc4.c &

### Wasm-specific micro-kernels
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MAX -D WASM=1 -o src/f32-rminmax/gen/f32-rmax-wasm-u1.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MAX -D WASM=1 -o src/f32-rminmax/gen/f32-rmax-wasm-u2-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MAX -D WASM=1 -o src/f32-rminmax/gen/f32-rmax-wasm-u3-acc3.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MAX -D WASM=1 -o src/f32-rminmax/gen/f32-rmax-wasm-u4-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MAX -D WASM=1 -o src/f32-rminmax/gen/f32-rmax-wasm-u4-acc4.c &

tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MIN -D WASM=1 -o src/f32-rminmax/gen/f32-rmin-wasm-u1.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MIN -D WASM=1 -o src/f32-rminmax/gen/f32-rmin-wasm-u2-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MIN -D WASM=1 -o src/f32-rminmax/gen/f32-rmin-wasm-u3-acc3.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MIN -D WASM=1 -o src/f32-rminmax/gen/f32-rmin-wasm-u4-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MIN -D WASM=1 -o src/f32-rminmax/gen/f32-rmin-wasm-u4-acc4.c &

tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=1 -D ACCUMULATORS=1 -D OP=MINMAX -D WASM=1 -o src/f32-rminmax/gen/f32-rminmax-wasm-u1.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=2 -D ACCUMULATORS=2 -D OP=MINMAX -D WASM=1 -o src/f32-rminmax/gen/f32-rminmax-wasm-u2-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=3 -D ACCUMULATORS=3 -D OP=MINMAX -D WASM=1 -o src/f32-rminmax/gen/f32-rminmax-wasm-u3-acc3.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=2 -D OP=MINMAX -D WASM=1 -o src/f32-rminmax/gen/f32-rminmax-wasm-u4-acc2.c &
tools/xngen src/f32-rminmax/scalar.c.in -D BATCH_TILE=4 -D ACCUMULATORS=4 -D OP=MINMAX -D WASM=1 -o src/f32-rminmax/gen/f32-rminmax-wasm-u4-acc4.c &

################################## Unit tests #################################
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f32-rmax.yaml --output test/f32-rmax.cc &
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f32-rmin.yaml --output test/f32-rmin.cc &
tools/generate-reduce-test.py --tester ReduceMicrokernelTester --spec test/f32-rminmax.yaml --output test/f32-rminmax.cc &

wait
