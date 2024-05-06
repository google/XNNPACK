#!/bin/sh
# Copyright 2023 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################### ARM NEONFP16ARITH ##################################
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-neonfp16arith-u8.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-neonfp16arith-u16-acc2.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-neonfp16arith-u24-acc3.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-neonfp16arith-u32-acc2.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-neonfp16arith-u32-acc4.c &

tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-neonfp16arith-u8.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-neonfp16arith-u16-acc2.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-neonfp16arith-u24-acc3.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-neonfp16arith-u32-acc2.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-neonfp16arith-u32-acc4.c &

tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=8  -D ACCUMULATORS=1 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-neonfp16arith-u8.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=16 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-neonfp16arith-u16-acc2.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=24 -D ACCUMULATORS=3 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-neonfp16arith-u24-acc3.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-neonfp16arith-u32-acc2.c &
tools/xngen src/f16-rminmax/neonfp16arith.c.in -D BATCH_TILE=32 -D ACCUMULATORS=4 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-neonfp16arith-u32-acc4.c &

################################## x86 AVX512FP16 #################################
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=32  -D ACCUMULATORS=1 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-avx512fp16-u32.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=64  -D ACCUMULATORS=2 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-avx512fp16-u64-acc2.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=96  -D ACCUMULATORS=3 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-avx512fp16-u96-acc3.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=128 -D ACCUMULATORS=2 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-avx512fp16-u128-acc2.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=128 -D ACCUMULATORS=4 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-avx512fp16-u128-acc4.c &

tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=32  -D ACCUMULATORS=1 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-avx512fp16-u32.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=64  -D ACCUMULATORS=2 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-avx512fp16-u64-acc2.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=96  -D ACCUMULATORS=3 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-avx512fp16-u96-acc3.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=128 -D ACCUMULATORS=2 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-avx512fp16-u128-acc2.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=128 -D ACCUMULATORS=4 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-avx512fp16-u128-acc4.c &

tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=32  -D ACCUMULATORS=1 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-avx512fp16-u32.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=64  -D ACCUMULATORS=2 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-avx512fp16-u64-acc2.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=96  -D ACCUMULATORS=3 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-avx512fp16-u96-acc3.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=128 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-avx512fp16-u128-acc2.c &
tools/xngen src/f16-rminmax/avx512fp16.c.in -D BATCH_TILE=128 -D ACCUMULATORS=4 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-avx512fp16-u128-acc4.c &

################################## x86 AVX512SKX #################################
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-avx512skx-u16.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-avx512skx-u32-acc2.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=48 -D ACCUMULATORS=3 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-avx512skx-u48-acc3.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-avx512skx-u64-acc2.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -D OP=MAX -o src/f16-rminmax/gen/f16-rmax-avx512skx-u64-acc4.c &

tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-avx512skx-u16.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-avx512skx-u32-acc2.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=48 -D ACCUMULATORS=3 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-avx512skx-u48-acc3.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-avx512skx-u64-acc2.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -D OP=MIN -o src/f16-rminmax/gen/f16-rmin-avx512skx-u64-acc4.c &

tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=16 -D ACCUMULATORS=1 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-avx512skx-u16.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=32 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-avx512skx-u32-acc2.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=48 -D ACCUMULATORS=3 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-avx512skx-u48-acc3.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=64 -D ACCUMULATORS=2 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-avx512skx-u64-acc2.c &
tools/xngen src/f16-rminmax/avx512skx.c.in -D BATCH_TILE=64 -D ACCUMULATORS=4 -D OP=MINMAX -o src/f16-rminmax/gen/f16-rminmax-avx512skx-u64-acc4.c &

#################################### Scalar ###################################
### Generic C micro-kernels
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=F16 -D ACCUMULATORS=1 -D OP=MINMAX -D WASM=0 -o src/f16-rminmax/gen/f16-rminmax-scalar-u1.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=F16 -D ACCUMULATORS=2 -D OP=MINMAX -D WASM=0 -o src/f16-rminmax/gen/f16-rminmax-scalar-u2-acc2.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=3 -D DATATYPE=F16 -D ACCUMULATORS=3 -D OP=MINMAX -D WASM=0 -o src/f16-rminmax/gen/f16-rminmax-scalar-u3-acc3.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=F16 -D ACCUMULATORS=2 -D OP=MINMAX -D WASM=0 -o src/f16-rminmax/gen/f16-rminmax-scalar-u4-acc2.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=F16 -D ACCUMULATORS=4 -D OP=MINMAX -D WASM=0 -o src/f16-rminmax/gen/f16-rminmax-scalar-u4-acc4.c &

tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=F16 -D ACCUMULATORS=1 -D OP=MAX -D WASM=0 -o src/f16-rminmax/gen/f16-rmax-scalar-u1.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=F16 -D ACCUMULATORS=2 -D OP=MAX -D WASM=0 -o src/f16-rminmax/gen/f16-rmax-scalar-u2-acc2.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=3 -D DATATYPE=F16 -D ACCUMULATORS=3 -D OP=MAX -D WASM=0 -o src/f16-rminmax/gen/f16-rmax-scalar-u3-acc3.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=F16 -D ACCUMULATORS=2 -D OP=MAX -D WASM=0 -o src/f16-rminmax/gen/f16-rmax-scalar-u4-acc2.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=F16 -D ACCUMULATORS=4 -D OP=MAX -D WASM=0 -o src/f16-rminmax/gen/f16-rmax-scalar-u4-acc4.c &

tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=1 -D DATATYPE=F16 -D ACCUMULATORS=1 -D OP=MIN -D WASM=0 -o src/f16-rminmax/gen/f16-rmin-scalar-u1.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=2 -D DATATYPE=F16 -D ACCUMULATORS=2 -D OP=MIN -D WASM=0 -o src/f16-rminmax/gen/f16-rmin-scalar-u2-acc2.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=3 -D DATATYPE=F16 -D ACCUMULATORS=3 -D OP=MIN -D WASM=0 -o src/f16-rminmax/gen/f16-rmin-scalar-u3-acc3.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=F16 -D ACCUMULATORS=2 -D OP=MIN -D WASM=0 -o src/f16-rminmax/gen/f16-rmin-scalar-u4-acc2.c &
tools/xngen src/f16-rminmax/scalar.c.in -D BATCH_TILE=4 -D DATATYPE=F16 -D ACCUMULATORS=4 -D OP=MIN -D WASM=0 -o src/f16-rminmax/gen/f16-rmin-scalar-u4-acc4.c &

wait
