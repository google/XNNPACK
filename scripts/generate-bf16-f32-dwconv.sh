#!/bin/sh
# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#################################### Scalar ###################################
tools/xngen src/bf16-f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-3p2c-minmax-scalar.c &
tools/xngen src/bf16-f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-4p2c-minmax-scalar.c &
tools/xngen src/bf16-f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-9p2c-minmax-scalar.c &
tools/xngen src/bf16-f32-dwconv/unipass-scalar.c.in -D CHANNEL_TILE=2 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-25p2c-minmax-scalar-acc2.c &

################################### ARM NEON ##################################
tools/xngen src/bf16-f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-3p8c-minmax-neonfma.c &
tools/xngen src/bf16-f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-3p16c-minmax-neonfma.c &
tools/xngen src/bf16-f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-4p8c-minmax-neonfma.c &
tools/xngen src/bf16-f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-4p16c-minmax-neonfma.c &
tools/xngen src/bf16-f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-9p8c-minmax-neonfma.c &
tools/xngen src/bf16-f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-9p16c-minmax-neonfma.c &
tools/xngen src/bf16-f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-25p8c-minmax-neonfma-acc2.c &
tools/xngen src/bf16-f32-dwconv/unipass-neon.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-25p16c-minmax-neonfma-acc2.c &

################################# ARM NEON BF16 ###############################
tools/xngen src/bf16-f32-dwconv/unipass-neonbf16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-3p8c-minmax-neonbf16.c &
tools/xngen src/bf16-f32-dwconv/unipass-neonbf16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-3p16c-minmax-neonbf16.c &
tools/xngen src/bf16-f32-dwconv/unipass-neonbf16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-4p8c-minmax-neonbf16.c &
tools/xngen src/bf16-f32-dwconv/unipass-neonbf16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-4p16c-minmax-neonbf16.c &
tools/xngen src/bf16-f32-dwconv/unipass-neonbf16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-9p8c-minmax-neonbf16.c &
tools/xngen src/bf16-f32-dwconv/unipass-neonbf16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-9p16c-minmax-neonbf16.c &
tools/xngen src/bf16-f32-dwconv/unipass-neonbf16.c.in -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-25p8c-minmax-neonbf16-acc2.c &
tools/xngen src/bf16-f32-dwconv/unipass-neonbf16.c.in -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-25p16c-minmax-neonbf16-acc2.c &

##################################### AVX2 ####################################
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx2 -D CHANNEL_TILE=8  -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-3p8c-minmax-avx2.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx2 -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-3p16c-minmax-avx2.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx2 -D CHANNEL_TILE=8  -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-4p8c-minmax-avx2.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx2 -D CHANNEL_TILE=16 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-4p16c-minmax-avx2.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx2 -D CHANNEL_TILE=8  -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-9p8c-minmax-avx2.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx2 -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-9p16c-minmax-avx2.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx2 -D CHANNEL_TILE=8  -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-25p8c-minmax-avx2-acc2.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx2 -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-25p16c-minmax-avx2-acc2.c &

################################### AVX512SKX #################################
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx512skx -D CHANNEL_TILE=16 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-3p16c-minmax-avx512skx.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx512skx -D CHANNEL_TILE=32 -D KERNEL_TILE=3  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-3p32c-minmax-avx512skx.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx512skx -D CHANNEL_TILE=16 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-4p16c-minmax-avx512skx.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx512skx -D CHANNEL_TILE=32 -D KERNEL_TILE=4  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-4p32c-minmax-avx512skx.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx512skx -D CHANNEL_TILE=16 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-9p16c-minmax-avx512skx.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx512skx -D CHANNEL_TILE=32 -D KERNEL_TILE=9  -D ACCUMULATORS=1 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-9p32c-minmax-avx512skx.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx512skx -D CHANNEL_TILE=16 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-25p16c-minmax-avx512skx-acc2.c &
tools/xngen src/bf16-f32-dwconv/unipass-simd.c.in -D ARCH=avx512skx -D CHANNEL_TILE=32 -D KERNEL_TILE=25 -D ACCUMULATORS=2 -o src/bf16-f32-dwconv/gen/bf16-f32-dwconv-25p32c-minmax-avx512skx-acc2.c &

wait
