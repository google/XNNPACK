#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=scalar -D SIMD_SIZE=1 -D CHANNELS=4 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-scalar.c &

################################### NEON ######################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=neon -D SIMD_SIZE=4 -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-c16.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=neon -D SIMD_SIZE=4 -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-c32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=neon -D SIMD_SIZE=4 -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-c64.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-rdsum/rvv.c.in -D ACCUMULATORS=7 LMUL=1 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-rvv-u1v.c &
tools/xngen src/f32-rdsum/rvv.c.in -D ACCUMULATORS=7 LMUL=2 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-rvv-u2v.c &
tools/xngen src/f32-rdsum/rvv.c.in -D ACCUMULATORS=7 LMUL=4 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-rvv-u4v.c &

#################################### SSE ######################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=sse2 -D SIMD_SIZE=4 -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse2-c16.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=sse2 -D SIMD_SIZE=4 -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse2-c32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=sse2 -D SIMD_SIZE=4 -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse2-c64.c &

#################################### AVX ######################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx -D SIMD_SIZE=8 -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c16.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx -D SIMD_SIZE=8 -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx -D SIMD_SIZE=8 -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-c64.c &

#################################### AVX512F ###################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx512f -D SIMD_SIZE=16 -D CHANNELS=16  -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-c16.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx512f -D SIMD_SIZE=16 -D CHANNELS=32  -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-c32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx512f -D SIMD_SIZE=16 -D CHANNELS=64  -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-c64.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx512f -D SIMD_SIZE=16 -D CHANNELS=128 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-c128.c &

################################## Hexagon HVX ################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=hvx -D SIMD_SIZE=32 -D CHANNELS=32  -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-hvx-c32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=hvx -D SIMD_SIZE=32 -D CHANNELS=64  -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-hvx-c64.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=hvx -D SIMD_SIZE=32 -D CHANNELS=128 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-hvx-c128.c &

#################################### WAsm SIMD ################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=wasmsimd -D SIMD_SIZE=4 -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-wasmsimd-c16.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=wasmsimd -D SIMD_SIZE=4 -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-wasmsimd-c32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=wasmsimd -D SIMD_SIZE=4 -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-wasmsimd-c64.c &

wait
