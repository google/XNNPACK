#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/qs8-rdsum/scalar.c.in -D ACCUMULATORS=7 -D REQUANTIZATION=FP32 -D VARIANT=FMAGIC -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-minmax-fp32-scalar-u1-acc1.c &

################################## ARM NEON ###################################
tools/xngen src/qs8-rdsum/neon.c.in -D CHANNELS=16  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-neon-c16.c &
tools/xngen src/qs8-rdsum/neon.c.in -D CHANNELS=32  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-neon-c32.c &
tools/xngen src/qs8-rdsum/neon.c.in -D CHANNELS=64  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-neon-c64.c &

################################### x86 SSE ###################################
tools/xngen src/qs8-rdsum/sse41.c.in -D CHANNELS=16  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-sse41-c16.c &
tools/xngen src/qs8-rdsum/sse41.c.in -D CHANNELS=32  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-sse41-c32.c &
tools/xngen src/qs8-rdsum/sse41.c.in -D CHANNELS=64  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-sse41-c64.c &

################################### x86 AVX2 ##################################
tools/xngen src/qs8-rdsum/avx2.c.in -D CHANNELS=32  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx2-c32.c &
tools/xngen src/qs8-rdsum/avx2.c.in -D CHANNELS=64  -D ACCUMULATORS=7 -D WASM=0 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx2-c64.c &

################################### x86 AVX512SKX ##################################
tools/xngen src/qs8-rdsum/avx512skx.c.in -D CHANNELS=64  -D ACCUMULATORS=7 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx512skx-c64.c &
tools/xngen src/qs8-rdsum/avx512skx.c.in -D CHANNELS=128  -D ACCUMULATORS=7 -o src/qs8-rdsum/gen/qs8-rdsum-7p7x-minmax-fp32-avx512skx-c128.c &

wait
