#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/f16-qs8-vcvt/neonfp16arith.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-neonfp16arith-u8.c &
tools/xngen src/f16-qs8-vcvt/neonfp16arith.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-neonfp16arith-u16.c &
tools/xngen src/f16-qs8-vcvt/neonfp16arith.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-neonfp16arith-u24.c &
tools/xngen src/f16-qs8-vcvt/neonfp16arith.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-neonfp16arith-u32.c &
tools/xngen src/f16-qs8-vcvt/neonfp16arith.c.in -D BATCH_TILE=64 -D DATATYPE=QS8 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-neonfp16arith-u64.c &

tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-neon-u8.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-neon-u16.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-neon-u24.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-neon-u32.c &

tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-neon-u8.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-neon-u16.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-neon-u24.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-neon-u32.c &

tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-neonv8-u8.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-neonv8-u16.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-neonv8-u24.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-neonv8-u32.c &

tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-neonv8-u8.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-neonv8-u16.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-neonv8-u24.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-neonv8-u32.c &

################################# x86 128-bit #################################
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-sse2-u8.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-sse2-u16.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-sse2-u24.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-sse2-u32.c &

tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-sse41-u8.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-sse41-u16.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-sse41-u24.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-sse41-u32.c &

tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-sse2-u8.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-sse2-u16.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-sse2-u24.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-sse2-u32.c &

################################# x86 256-bit #################################
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx-u8.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx-u16.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx-u24.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx-u32.c &

tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx-u8.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx-u16.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx-u24.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx-u32.c &

tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx2-u16.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx2-u32.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=48 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx2-u48.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx2-u64.c &

tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx2-u16.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx2-u32.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=48 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx2-u48.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx2-u64.c &

################################# x86 512-bit #################################
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=32  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx512skx-u32.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=64  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx512skx-u64.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=96  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx512skx-u96.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=128 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-avx512skx-u128.c &

tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=32  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx512skx-u32.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=64  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx512skx-u64.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=96  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx512skx-u96.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=128 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-avx512skx-u128.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasmsimd-cvt-u8.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasmsimd-cvt-u16.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasmsimd-cvt-u24.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasmsimd-cvt-u32.c &

tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasmsimd-cvt-u8.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasmsimd-cvt-u16.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasmsimd-cvt-u24.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasmsimd-cvt-u32.c &

tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasmsimd-magic-u8.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasmsimd-magic-u16.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasmsimd-magic-u24.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasmsimd-magic-u32.c &

tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasmsimd-magic-u8.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasmsimd-magic-u16.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasmsimd-magic-u24.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasmsimd-magic-u32.c &

##################################### WAsm ####################################
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=1 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=1 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasm-fmagic-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=2 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=1 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasm-fmagic-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=3 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=1 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasm-fmagic-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=4 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=1 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-wasm-fmagic-u4.c &

tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=1 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=1 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasm-fmagic-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=2 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=1 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasm-fmagic-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=3 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=1 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasm-fmagic-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=4 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=1 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-wasm-fmagic-u4.c &

################################## Hexagon HVX ###################################
tools/xngen src/f32-qs8-vcvt/hvx.c.in -D BATCH_TILE=32  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-hvx-u32.c &
tools/xngen src/f32-qs8-vcvt/hvx.c.in -D BATCH_TILE=64  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-hvx-u64.c &
tools/xngen src/f32-qs8-vcvt/hvx.c.in -D BATCH_TILE=96  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-hvx-u96.c &
tools/xngen src/f32-qs8-vcvt/hvx.c.in -D BATCH_TILE=128  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-hvx-u128.c &
tools/xngen src/f32-qs8-vcvt/hvx.c.in -D BATCH_TILE=256  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-hvx-u256.c &

#################################### Scalar ###################################
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=1 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-fmagic-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=2 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-fmagic-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=3 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-fmagic-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=4 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-fmagic-u4.c &

tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=1 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-fmagic-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=2 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-fmagic-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=3 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-fmagic-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=4 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-fmagic-u4.c &

tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=1 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-fmagic-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=2 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-fmagic-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=3 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-fmagic-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=4 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-fmagic-u4.c &

tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=1 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-imagic-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=2 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-imagic-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=3 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-imagic-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=4 -D IDATATYPE=F16 -D ODATATYPE=QS8 -D WASM=0 -o src/f16-qs8-vcvt/gen/f16-qs8-vcvt-scalar-imagic-u4.c &

tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=1 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-imagic-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=2 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-imagic-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=3 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-imagic-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=4 -D IDATATYPE=F32 -D ODATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-imagic-u4.c &

tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=1 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-imagic-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=2 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-imagic-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=3 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-imagic-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=4 -D IDATATYPE=F32 -D ODATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-imagic-u4.c &

tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-lrintf-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-lrintf-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=3 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-lrintf-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/f32-qs8-vcvt-scalar-lrintf-u4.c &

tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-lrintf-u1.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-lrintf-u2.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=3 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-lrintf-u3.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/f32-qu8-vcvt-scalar-lrintf-u4.c &

wait
