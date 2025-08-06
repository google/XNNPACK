#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## Scalar #####################################
tools/xngen src/qs8-rdsum/scalar.c.in -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-minmax-fp32-scalar-u1-acc1.c &

################################## ARM NEON ###################################
tools/xngen src/qs8-rdsum/neon.c.in -D CHANNELS=16  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-neon-u16.c &
tools/xngen src/qs8-rdsum/neon.c.in -D CHANNELS=32  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-neon-u32.c &
tools/xngen src/qs8-rdsum/neon.c.in -D CHANNELS=64  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-neon-u64.c &

################################## x86 SSE ####################################
tools/xngen src/qs8-rdsum/sse41.c.in -D CHANNELS=16  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-sse41-u16.c &
tools/xngen src/qs8-rdsum/sse41.c.in -D CHANNELS=32  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-sse41-u32.c &
tools/xngen src/qs8-rdsum/sse41.c.in -D CHANNELS=64  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-sse41-u64.c &

################################## x86 AVX2 ###################################
tools/xngen src/qs8-rdsum/avx2.c.in -D CHANNELS=32  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx2-u32.c &
tools/xngen src/qs8-rdsum/avx2.c.in -D CHANNELS=64  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx2-u64.c &

################################## x86 AVX512SKX ##############################
tools/xngen src/qs8-rdsum/avx512skx.c.in -D CHANNELS=64  -D ACCUMULATORS=7 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx512skx-u64.c &
tools/xngen src/qs8-rdsum/avx512skx.c.in -D CHANNELS=128  -D ACCUMULATORS=7 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx512skx-u128.c &

################################## Wasm SIMD ##################################
tools/xngen src/qs8-rdsum/wasmsimd.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -D DATATYPE=QS8 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-wasmsimd-u16.c &
tools/xngen src/qs8-rdsum/wasmsimd.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -D DATATYPE=QS8 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-wasmsimd-u32.c &
tools/xngen src/qs8-rdsum/wasmsimd.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -D DATATYPE=QS8 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-wasmsimd-u64.c &

################################## Wasm SIMD ##################################
tools/xngen src/qs8-rdsum/rvv.c.in -D LMUL=1 -D ACCUMULATORS=7 -D DATATYPE=QS8 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-rvv-u1v.c &
tools/xngen src/qs8-rdsum/rvv.c.in -D LMUL=2 -D ACCUMULATORS=7 -D DATATYPE=QS8 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-rvv-u2v.c &

wait
