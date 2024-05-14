#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-rdsum/scalar.c.in -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-scalar.c &

################################### NEON ######################################
tools/xngen src/f32-rdsum/neon.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-c16.c &
tools/xngen src/f32-rdsum/neon.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-c32.c &
tools/xngen src/f32-rdsum/neon.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-c64.c &

#################################### SSE ######################################
tools/xngen src/f32-rdsum/sse.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse-c16.c &
tools/xngen src/f32-rdsum/sse.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse-c32.c &
tools/xngen src/f32-rdsum/sse.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse-c64.c &

#################################### AVX ######################################
tools/xngen src/f32-rdsum/avx.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c16.c &
tools/xngen src/f32-rdsum/avx.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c32.c &
tools/xngen src/f32-rdsum/avx.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c64.c &

#################################### AVX512F ###################################
tools/xngen src/f32-rdsum/avx512.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-c16.c &
tools/xngen src/f32-rdsum/avx512.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-c32.c &
tools/xngen src/f32-rdsum/avx512.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-c64.c &
tools/xngen src/f32-rdsum/avx512.c.in -D CHANNELS=128 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-c128.c &

#################################### WAsm SIMD ################################
tools/xngen src/f32-rdsum/wasm-simd.c.in -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-wasmsimd-c16.c &
tools/xngen src/f32-rdsum/wasm-simd.c.in -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-wasmsimd-c32.c &
tools/xngen src/f32-rdsum/wasm-simd.c.in -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-wasmsimd-c64.c &

wait
