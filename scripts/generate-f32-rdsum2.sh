#!/bin/sh
# Copyright 2024-2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

################################# SIMD Wrappers ################################
tools/xngen src/f32-rdsum2/simd.c.in -D ARCH=scalar -D SIMD_SIZE=1 -D CHANNELS=4 -D ACCUMULATORS=7 -o src/f32-rdsum2/gen/f32-rdsum2-7p7x-minmax-scalar.c &
tools/xngen src/f32-rdsum2/simd.c.in -D ARCH=neon -D SIMD_SIZE=4 -D CHANNELS=16,32,64 -D ACCUMULATORS=7 -o src/f32-rdsum2/gen/f32-rdsum2-7p7x-minmax-neon.c &
tools/xngen src/f32-rdsum2/simd.c.in -D ARCH=sse2 -D SIMD_SIZE=4 -D CHANNELS=16,32,64 -D ACCUMULATORS=7 -o src/f32-rdsum2/gen/f32-rdsum2-7p7x-minmax-sse2.c &
tools/xngen src/f32-rdsum2/simd.c.in -D ARCH=avx -D SIMD_SIZE=8 -D CHANNELS=16,32,64 -D ACCUMULATORS=7 -o src/f32-rdsum2/gen/f32-rdsum2-7p7x-minmax-avx.c &
tools/xngen src/f32-rdsum2/simd.c.in -D ARCH=avx512f -D SIMD_SIZE=16 -D CHANNELS=16,32,64,128 -D ACCUMULATORS=7 -o src/f32-rdsum2/gen/f32-rdsum2-7p7x-minmax-avx512f.c &
tools/xngen src/f32-rdsum2/simd.c.in -D ARCH=hvx -D SIMD_SIZE=32 -D CHANNELS=32,64,128 -D ACCUMULATORS=7 -o src/f32-rdsum2/gen/f32-rdsum2-7p7x-minmax-hvx.c &
tools/xngen src/f32-rdsum2/simd.c.in -D ARCH=wasmsimd -D SIMD_SIZE=4 -D CHANNELS=16,32,64 -D ACCUMULATORS=7 -o src/f32-rdsum2/gen/f32-rdsum2-7p7x-minmax-wasmsimd.c &

wait
