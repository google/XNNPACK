#!/bin/sh
# Copyright 2021 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################## ARM NEON ###################################
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neon-x8.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neon-x16.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neon-x24.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neon-x32.c &

tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neon-x8.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neon-x16.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neon-x24.c &
tools/xngen src/f32-qs8-vcvt/neon.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neon-x32.c &

tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neonv8-x8.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neonv8-x16.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neonv8-x24.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-neonv8-x32.c &

tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neonv8-x8.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neonv8-x16.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neonv8-x24.c &
tools/xngen src/f32-qs8-vcvt/neonv8.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-neonv8-x32.c &

################################# x86 128-bit #################################
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse2-x8.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse2-x16.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse2-x24.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse2-x32.c &

tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse41-x8.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse41-x16.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse41-x24.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=4 -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-sse41-x32.c &

tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-sse2-x8.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-sse2-x16.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-sse2-x24.c &
tools/xngen src/f32-qs8-vcvt/sse.c.in -D SSE=2 -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-sse2-x32.c &

################################# x86 256-bit #################################
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx-x8.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx-x16.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx-x24.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx-x32.c &

tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx-x8.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx-x16.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx-x24.c &
tools/xngen src/f32-qs8-vcvt/avx.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx-x32.c &

tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx2-x16.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx2-x32.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=48 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx2-x48.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx2-x64.c &

tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx2-x16.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx2-x32.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=48 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx2-x48.c &
tools/xngen src/f32-qs8-vcvt/avx2.c.in -D BATCH_TILE=64 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx2-x64.c &

################################# x86 512-bit #################################
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=32  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx512skx-x32.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=64  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx512skx-x64.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=96  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx512skx-x96.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=128 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-avx512skx-x128.c &

tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=32  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx512skx-x32.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=64  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx512skx-x64.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=96  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx512skx-x96.c &
tools/xngen src/f32-qs8-vcvt/avx512skx.c.in -D BATCH_TILE=128 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-avx512skx-x128.c &

################################## WAsm SIMD ##################################
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-wasmsimd-cvt-x8.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-wasmsimd-cvt-x16.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-wasmsimd-cvt-x24.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-wasmsimd-cvt-x32.c &

tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-wasmsimd-cvt-x8.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-wasmsimd-cvt-x16.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-wasmsimd-cvt-x24.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-cvt.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-wasmsimd-cvt-x32.c &

tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=8  -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-wasmsimd-magic-x8.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=16 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-wasmsimd-magic-x16.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=24 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-wasmsimd-magic-x24.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=32 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-wasmsimd-magic-x32.c &

tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=8  -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-wasmsimd-magic-x8.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=16 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-wasmsimd-magic-x16.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=24 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-wasmsimd-magic-x24.c &
tools/xngen src/f32-qs8-vcvt/wasmsimd-magic.c.in -D BATCH_TILE=32 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-wasmsimd-magic-x32.c &

##################################### WAsm ####################################
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -D WASM=1 -o src/f32-qs8-vcvt/gen/vcvt-wasm-fmagic-x1.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -D WASM=1 -o src/f32-qs8-vcvt/gen/vcvt-wasm-fmagic-x2.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=3 -D DATATYPE=QS8 -D WASM=1 -o src/f32-qs8-vcvt/gen/vcvt-wasm-fmagic-x3.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -D WASM=1 -o src/f32-qs8-vcvt/gen/vcvt-wasm-fmagic-x4.c &

tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -D WASM=1 -o src/f32-qu8-vcvt/gen/vcvt-wasm-fmagic-x1.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -D WASM=1 -o src/f32-qu8-vcvt/gen/vcvt-wasm-fmagic-x2.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=3 -D DATATYPE=QU8 -D WASM=1 -o src/f32-qu8-vcvt/gen/vcvt-wasm-fmagic-x3.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -D WASM=1 -o src/f32-qu8-vcvt/gen/vcvt-wasm-fmagic-x4.c &

#################################### Scalar ###################################
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/vcvt-scalar-fmagic-x1.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/vcvt-scalar-fmagic-x2.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=3 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/vcvt-scalar-fmagic-x3.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/vcvt-scalar-fmagic-x4.c &

tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/vcvt-scalar-fmagic-x1.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/vcvt-scalar-fmagic-x2.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=3 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/vcvt-scalar-fmagic-x3.c &
tools/xngen src/f32-qs8-vcvt/scalar-fmagic.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/vcvt-scalar-fmagic-x4.c &

tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-scalar-imagic-x1.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-scalar-imagic-x2.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=3 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-scalar-imagic-x3.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -o src/f32-qs8-vcvt/gen/vcvt-scalar-imagic-x4.c &

tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-scalar-imagic-x1.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-scalar-imagic-x2.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=3 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-scalar-imagic-x3.c &
tools/xngen src/f32-qs8-vcvt/scalar-imagic.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -o src/f32-qu8-vcvt/gen/vcvt-scalar-imagic-x4.c &

tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=1 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/vcvt-scalar-lrintf-x1.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=2 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/vcvt-scalar-lrintf-x2.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=3 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/vcvt-scalar-lrintf-x3.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=4 -D DATATYPE=QS8 -D WASM=0 -o src/f32-qs8-vcvt/gen/vcvt-scalar-lrintf-x4.c &

tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=1 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/vcvt-scalar-lrintf-x1.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=2 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/vcvt-scalar-lrintf-x2.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=3 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/vcvt-scalar-lrintf-x3.c &
tools/xngen src/f32-qs8-vcvt/scalar-lrintf.c.in -D BATCH_TILE=4 -D DATATYPE=QU8 -D WASM=0 -o src/f32-qu8-vcvt/gen/vcvt-scalar-lrintf-x4.c &

################################## Unit tests #################################
tools/generate-vcvt-test.py --spec test/f32-qs8-vcvt.yaml --output test/f32-qs8-vcvt.cc &
tools/generate-vcvt-test.py --spec test/f32-qu8-vcvt.yaml --output test/f32-qu8-vcvt.cc &

wait
