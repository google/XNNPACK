#!/bin/sh
# Copyright 2024 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=scalar -D SIMD_SIZE=1 -D CHANNELS=4 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-scalar.c &

################################### NEON ######################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=neon -D SIMD_SIZE=4 -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-u16.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=neon -D SIMD_SIZE=4 -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-u32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=neon -D SIMD_SIZE=4 -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-neon-u64.c &

################################ RISC-V Vector ################################
tools/xngen src/f32-rdsum/rvv.c.in -D ACCUMULATORS=7 LMUL=1 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-rvv-u1v.c &
tools/xngen src/f32-rdsum/rvv.c.in -D ACCUMULATORS=7 LMUL=2 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-rvv-u2v.c &
tools/xngen src/f32-rdsum/rvv.c.in -D ACCUMULATORS=7 LMUL=4 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-rvv-u4v.c &

#################################### SSE ######################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=sse2 -D SIMD_SIZE=4 -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse2-u16.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=sse2 -D SIMD_SIZE=4 -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse2-u32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=sse2 -D SIMD_SIZE=4 -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-sse2-u64.c &

#################################### AVX ######################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx -D SIMD_SIZE=8 -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-u16.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx -D SIMD_SIZE=8 -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-u32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx -D SIMD_SIZE=8 -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx-u64.c &

#################################### AVX512F ###################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx512f -D SIMD_SIZE=16 -D CHANNELS=16  -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-u16.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx512f -D SIMD_SIZE=16 -D CHANNELS=32  -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-u32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx512f -D SIMD_SIZE=16 -D CHANNELS=64  -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-u64.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=avx512f -D SIMD_SIZE=16 -D CHANNELS=128 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-avx512f-u128.c &

################################## Hexagon HVX ################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=hvx -D SIMD_SIZE=32 -D CHANNELS=32  -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-hvx-u32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=hvx -D SIMD_SIZE=32 -D CHANNELS=64  -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-hvx-u64.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=hvx -D SIMD_SIZE=32 -D CHANNELS=128 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-hvx-u128.c &

#################################### WAsm SIMD ################################
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=wasmsimd -D SIMD_SIZE=4 -D CHANNELS=16 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-wasmsimd-u16.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=wasmsimd -D SIMD_SIZE=4 -D CHANNELS=32 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-wasmsimd-u32.c &
tools/xngen src/f32-rdsum/simd.c.in -D ARCH=wasmsimd -D SIMD_SIZE=4 -D CHANNELS=64 -D ACCUMULATORS=7 -o src/f32-rdsum/gen/f32-rdsum-7p7x-minmax-wasmsimd-u64.c &

wait
